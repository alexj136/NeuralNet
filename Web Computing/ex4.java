import javax.xml.parsers.*;
import javax.xml.transform.*;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.dom.DOMSource;
import org.w3c.dom.*;
import java.io.File;

class ex4 {
	public static void main ( String[] args ) throws Exception {
        new ex4().doPart1 ( "ex1q1-2.xml" );
	}

    /**
     * Create an DOM Document object from an XML file
     */
    public Document fromFile ( String fileName ) throws Exception {
		return DocumentBuilderFactory.newInstance()
            .newDocumentBuilder()
            .parse( new File ( fileName ) );
    }

    /**
     * Write an XML file from a DOM Document object
     */
    public void toFile ( String fileName ) throws Exception {
        Transformer trans = TransformerFactory.newInstance().newTransformer();
        StreamResult out = new StreamResult ( new File ( fileName ) );
        trans.transform ( new DOMSource ( doc ), out );
    }

    /**
     * Asserts that the given filename ends with ".xml"
     */
    public void assertXMLExtension ( String fileName ) throws Exception {
        int len = fileName.length();
        assert ( ".xml".equals ( fileName.substring ( len - 5, len - 1 ) ) );
    }

    /**
     * Asserts that the given filename ends with ".xml" and returns the string
     * that appears before the .xml extension.
     */
    public void baseName ( String fileName ) throws Exception {
        assertXMLExtension ( fileName );
        int len = fileName.length();
        return fileName.substring ( 0, len - 4 );
    }

    public void doPart1 ( String fileName ) throws Exception {
        // Assert that the passed file has a .xml extension
        int len = fileName.length();
        assert ( ".xml".equals ( fileName.substring ( len - 5, len - 1 ) ) );

        // Obtain an XML DOM object from the given file
        Document doc = fromFile ( fileName );

        // Obtain a reference to the document's root node
		Node root = doc.getDocumentElement();

        // Reverse all children from the root node
		recRevChildren ( root );

        // Write the new XML tree to a new file
        toFile ( "REV" + fileName );
    }

	public void recRevChildren ( Node node ) {
		NodeList children = node.getChildNodes();

		for(int i = children.getLength() - 1; i >= 0; i--) {
            Node curChild = children.item ( i );
			recRevChildren ( curChild );
			node.removeChild ( curChild );
			node.appendChild ( curChild );
		}
	}

    public void doPart2 ( String fileName1, String fileName2 ) throws Exception {

        Document doc1 = fromFile ( fileName1 );
        Document doc2 = fromFile ( fileName2 );

        String newRootName = baseName ( fileName1 )
                  + "PLUS" + baseName ( fileName2 );

        Element newRoot = doc1.createElement ( newRootName );
        // WAS HERE
    }

    public void doPart3() throws Exception {
        Document doc = fromFile ( "ex1q1-2.xml" );
        Node root = (Node) Document.getDocumentElement( doc );
        recRemoveOfficialInfo ( root );
        toFile ( doc );
    }

    public void recRemoveOfficialInfo ( Node startNode ) {
        // WAS ALSO HERE
    }
}
