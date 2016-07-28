import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.tartarus.snowball.SnowballStemmer;
import org.tartarus.snowball.ext.englishStemmer;

import com.fasterxml.jackson.databind.ObjectMapper;

import edu.stanford.nlp.ling.CyclicCoreLabel;
import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.parser.lexparser.LexicalizedParser;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;
import edu.stanford.nlp.trees.GrammaticalStructure;
import edu.stanford.nlp.trees.GrammaticalStructureFactory;
import edu.stanford.nlp.trees.PennTreebankLanguagePack;
import edu.stanford.nlp.trees.Tree;
import edu.stanford.nlp.trees.TreebankLanguagePack;
import edu.stanford.nlp.trees.TypedDependency;

class Tagger {
	private MaxentTagger tagger;
	private SnowballStemmer stemmer;
	private LexicalizedParser lp;
	private GrammaticalStructureFactory gsf;

	private Tagger(
		MaxentTagger tagger,
		SnowballStemmer stemmer,
		LexicalizedParser lp,
		GrammaticalStructureFactory gsf)
	{
		this.tagger = tagger;
		this.stemmer = stemmer;
		this.lp = lp;
		this.gsf = gsf;
	}

	public static void main(String[] args) throws Exception {
		Tagger tagger = init();
		
		String cmd = args[0];
		String inFile = args[1];
		String outFile = args[2];

		System.out.println("Command: " + cmd + " inFile=" + inFile + " outFile=" + outFile);
		
		if ("train".equals(cmd)) {
			tagger.convertTrain(inFile, outFile);
		} else if ("test".equals(cmd)) {
			tagger.convertTest(inFile, outFile);
		}
	}

	private static Tagger init() throws Exception {
		MaxentTagger tagger = new MaxentTagger("dep/wsj-0-18-left3words.tagger");

		SnowballStemmer stemmer = new englishStemmer();

		LexicalizedParser lp = LexicalizedParser.loadModel(
				"edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz",
				"-maxLength", "80", "-retainTmpSubcategories");

		TreebankLanguagePack tlp = new PennTreebankLanguagePack();
		GrammaticalStructureFactory gsf = tlp.grammaticalStructureFactory();

		return new Tagger(tagger, stemmer, lp, gsf);
	}

	public void convertTest(String fileName, String outFile) throws Exception {
		BufferedReader in = new BufferedReader(new FileReader(fileName));
		BufferedWriter out = new BufferedWriter(new FileWriter(outFile));
		String line;
		out.write(in.readLine()); // header
		out.newLine();
		out.flush();
		
		System.out.println("Process test.");
		
		int i = 0;
		while (null != (line = in.readLine())) {
			if (0 == ++i % 200) {
				System.out.println("  processed " + i + " lines.");
			}
			
			String fields[] = line.split(",", 2);
			String dt = fields[0];
			String rawText = fields[1];

			Row row = parse(rawText);

			ObjectMapper mapper = new ObjectMapper();
			out.write(dt + "," + mapper.writeValueAsString(row) + "\n");
			out.flush();
		}
		out.close();
		in.close();
	}

	public void convertTrain(String fileName, String outFile) throws Exception {
		BufferedReader in = new BufferedReader(new FileReader(fileName));
		BufferedWriter out = new BufferedWriter(new FileWriter(outFile));
		String line;

		out.write(in.readLine()); // header
		out.newLine();
		out.flush();

		System.out.println("Process train.");

		int i = 0;
		while (null != (line = in.readLine())) {
			if (0 == ++i % 200) {
				System.out.println("  processed " + i + " lines.");
			}
			
			String fields[] = line.split(",", 3);
			String id = fields[0];
			String rawText = fields[2];
			Row row = parse(rawText);

			ObjectMapper mapper = new ObjectMapper();
			out.write(id + "," + mapper.writeValueAsString(row) + "\n");
			out.flush();
		}
		out.close();
		in.close();
	}

	private Row parse(String text) {
		String normText = norm(text);
		List<List<HasWord>> taggedSentences = MaxentTagger.tokenizeText(new StringReader(normText));

		Row row = new Row();
		row.rawText = text;
		row.text = normText;

		for (List<HasWord> taggedSentence : taggedSentences) {
			Sen sentence = new Sen();
			for (TaggedWord tok : tagger.tagSentence(taggedSentence)) {
				sentence.tokens.add(
					new Token(tok.word(), tok.tag(), stem(stemmer, tok.word())));
			}

			if (taggedSentence.size() < 80)
			{
				Tree parse = lp.apply(taggedSentence);
				GrammaticalStructure gs = gsf.newGrammaticalStructure(parse);
				for (TypedDependency dep : gs.typedDependenciesCCprocessed()) {
					sentence.dependencies.add(
						new Dep(
							dep.reln().toString(),
							toDepWord(dep.gov().label()),
							toDepWord(dep.dep().label())));
				}
			} else {
				System.err.println("long sentence: " + taggedSentence);
			}

			row.sentences.add(sentence);
		}
		return row;
	}

	private DepWord toDepWord(CyclicCoreLabel label) {
		return (0 == label.index())
			? new DepWord("$ROOT", "$ROOT", 0)
			: new DepWord(label.value(), stem(stemmer, label.value()), label.index());
	}

	public static String norm(String s) {
		// strip('"')
		while (s.startsWith("\""))
			s = s.substring(1);
		while (s.endsWith("\""))
			s = s.substring(0, s.length() - 1);

		s = s.replaceAll("</?[a-zA-Z][^>]*>", " ");
		s = s.replaceAll("\\\\x[0-9a-zA-Z]{1,10}", " ");
		s = s.replaceAll("\\\\x[0-9a-zA-Z]{1,10}", " ");
		s = s.replaceAll("&[a-zA-Z]{1,10};", " ");
		s = s.replaceAll("&#[0-9a-zA-Z];", " ");
		s = s.replaceAll("\\\\[nrt]", " ");

		return s.trim();
	}

	public static String stem(SnowballStemmer stemmer, String word) {
		stemmer.setCurrent(word);
		stemmer.stem();
		return stemmer.getCurrent();
	}
}

class Row {
	public List<Sen> sentences = new ArrayList<Sen>();
	public String rawText = "";
	public String text = "";
}

class Sen {
	public List<Token> tokens = new ArrayList<Token>();
	public List<Dep> dependencies = new ArrayList<Dep>();
}

class Token {
	public Token(String word, String tag, String stem) {
		this.word = word;
		this.tag = tag;
		this.stem = stem;
	}

	public String word;
	public String tag;
	public String stem;
}

class Dep {
	public Dep(String rel, DepWord gov, DepWord dep) {
		this.rel = rel;
		this.gov = gov;
		this.dep = dep;
	}

	public String rel;

	public DepWord gov;
	public DepWord dep;
}

class DepWord {
	public DepWord(String word, String stem, int index) {
		this.word = word;
		this.stem = stem;
		this.index = index;
	}

	public String word;
	public String stem;
	public int index;
}
