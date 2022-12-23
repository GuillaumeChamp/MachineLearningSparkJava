package org.example;

import org.codehaus.janino.Java;

import java.io.IOException;
import java.util.logging.FileHandler;
import java.util.logging.Level;
import java.util.logging.Logger;
import java.util.logging.SimpleFormatter;

public class MyLog {
    private static java.util.logging.Logger log;
    public static void log(String s){
        log.fine(s);
        System.out.println(s);
    }
    public static void init(String outPath) throws IOException {
        FileHandler fh = new FileHandler(outPath+"MyLogFile.log", false);
        Logger log = Logger.getLogger(MyLog.class.getName());
        log.setLevel(Level.FINE);
        log.addHandler(fh);
        SimpleFormatter formatter = new SimpleFormatter();
        fh.setFormatter(formatter);
        MyLog.log = log;
    }
}
