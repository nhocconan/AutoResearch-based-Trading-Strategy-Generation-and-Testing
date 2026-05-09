#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_RSI_Chop_Filter_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA, RSI, chop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate KAMA on daily
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if np.sum(volatility[i-9:i+1]) > 0:
            er[i] = np.sum(change[i-9:i+1]) / np.sum(volatility[i-9:i+1])
        else:
            er[i] = 0
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama = kama.astype(float)
    kama[:9] = np.nan
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI(14) on daily
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi[:13] = np.nan
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate Choppiness Index(14) on daily
    atr = np.zeros_like(close_1d)
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(14, min_periods=14).mean().values
    
    max_high = pd.Series(high).rolling(14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(14, min_periods=14).min().values
    chop = np.where((max_high - min_low) > 0, 
                    100 * np.log10(atr.sum() / (max_high - min_low)) / np.log10(14), 
                    50)
    chop[:13] = np.nan
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume confirmation: current volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(20, min_periods=20).mean().values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        kama_val = kama_aligned[i]
        rsi_val = rsi_aligned[i]
        chop_val = chop_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price > KAMA, RSI < 40 (oversold), chop > 61.8 (ranging)
            if close[i] > kama_val and rsi_val < 40 and chop_val > 61.8 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA, RSI > 60 (overbought), chop > 61.8 (ranging)
            elif close[i] < kama_val and rsi_val > 60 and chop_val > 61.8 and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA or RSI > 70
            if close[i] < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA or RSI < 30
            if close[i] > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals