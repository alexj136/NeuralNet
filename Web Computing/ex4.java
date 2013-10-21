import javax.xml.parsers.*;
import org.w3c.dom.*;
import java.io.File;

class ex4 {
	public static void main ( String[] args ) throws Exception {
		// Exercise 1
		DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
		DocumentBuilder db = dbf.newDocumentBuilder();
		Document doc = db.parse( new File ( "ex1q1-2.xml" ) );
		Node root = doc.getDocumentElement();
		recRevChildren ( root );
	}

	public void recRevChildren ( Node node ) {
		NodeList children = node.getChildNodes();

		for(int i = 0; i < children.getLength(); i++) {
			recRevChildren ( children.item ( i ) );
			node.removeChild ( children.item ( i ) );
		}

		for(int i = children.getLength() - 1; i >= 0; i++) {
			node.appendChild ( children.item ( i ) );
		}
	}
}
