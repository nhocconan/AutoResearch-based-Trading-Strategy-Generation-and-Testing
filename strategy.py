#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Choppiness Index regime filter with 1-day KAMA trend and volume confirmation
# Long when KAMA(1d) is rising (bullish trend) AND Choppiness Index(12h) < 38.2 (trending regime) AND volume > 1.5x 20-period average
# Short when KAMA(1d) is falling (bearish trend) AND Choppiness Index(12h) < 38.2 (trending regime) AND volume > 1.5x 20-period average
# Exit when KAMA trend reverses OR Choppiness Index > 61.8 (range regime)
# This avoids whipsaws in ranging markets while capturing strong trends with volume confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for KAMA trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily
    # ER = Efficiency Ratio, SC = Smoothing Constant
    change = np.abs(np.diff(df_1d['close'], prepend=df_1d['close'][0]))
    volatility = np.abs(np.diff(df_1d['close']))
    er = change / (volatility + 1e-10)  # Avoid division by zero
    sc = (er * (2/3 - 2/30) + 2/30) ** 2  # Fast=2, Slow=30
    kama = np.zeros_like(df_1d['close'])
    kama[0] = df_1d['close'][0]
    for i in range(1, len(df_1d)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'][i] - kama[i-1])
    kama = kama
    
    # Calculate Choppiness Index on 12h
    # True Range = max(high-low, abs(high-previous close), abs(low-previous close))
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10(atr_sum / (highest_high - lowest_low)) / log10(14)
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # Load 12h data for alignment
    df_12h = get_htf_data(prices, '12h')
    
    # Align KAMA trend (rising/falling) to 12h timeframe
    kama_rising = np.diff(kama, prepend=0) > 0  # True when KAMA is rising
    kama_rising = kama_rising.astype(float)
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising)
    
    # Align Choppiness Index to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or np.isnan(kama_rising_aligned[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: KAMA rising + trending regime (Chop < 38.2) + volume confirmation
            if (kama_rising_aligned[i] > 0.5 and chop_aligned[i] < 38.2 and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: KAMA falling + trending regime (Chop < 38.2) + volume confirmation
            elif (kama_rising_aligned[i] < 0.5 and chop_aligned[i] < 38.2 and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns bearish OR Chop enters range regime (Chop > 61.8)
            if (kama_rising_aligned[i] < 0.5) or (chop_aligned[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: KAMA turns bullish OR Chop enters range regime (Chop > 61.8)
            if (kama_rising_aligned[i] > 0.5) or (chop_aligned[i] > 61.8):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_KAMA_Chop_Volume"
timeframe = "12h"
leverage = 1.0