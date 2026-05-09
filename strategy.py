#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyKeltner_Breakout_Squeeze"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly ATR and EMA for Keltner Channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly True Range
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    tr_w1 = high_w[1:] - low_w[:-1]
    tr_w2 = np.abs(high_w[1:] - close_w[:-1])
    tr_w3 = np.abs(low_w[:-1] - close_w[:-1])
    tr_w = np.concatenate([[np.max([tr_w1[0], tr_w2[0], tr_w3[0]])], np.maximum(tr_w1, np.maximum(tr_w2, tr_w3))])
    atr_w20 = pd.Series(tr_w).rolling(window=20, min_periods=20).mean().values
    
    # Weekly EMA20 for Keltner middle
    ema20_w = pd.Series(close_w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Weekly Keltner Bands
    upper_w = ema20_w + 2.0 * atr_w20
    lower_w = ema20_w - 2.0 * atr_w20
    
    # Align weekly Keltner bands to daily
    upper_w_aligned = align_htf_to_ltf(prices, df_1w, upper_w)
    lower_w_aligned = align_htf_to_ltf(prices, df_1w, lower_w)
    ema20_w_aligned = align_htf_to_ltf(prices, df_1w, ema20_w)
    
    # Daily Bollinger Bands (20, 2.0) for squeeze detection
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma20 + 2.0 * std20
    lower_bb = sma20 - 2.0 * std20
    
    # Bollinger Band Width for squeeze
    bb_width = (upper_bb - lower_bb) / sma20
    bb_width_ma50 = pd.Series(bb_width).rolling(window=50, min_periods=50).mean().values
    squeeze = bb_width < 0.5 * bb_width_ma50  # Bollinger squeeze condition
    
    # Volume confirmation: 20-day volume average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_surge = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(upper_w_aligned[i]) or np.isnan(lower_w_aligned[i]) or \
           np.isnan(ema20_w_aligned[i]) or np.isnan(squeeze[i]) or np.isnan(vol_surge[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above weekly Keltner upper band during Bollinger squeeze + volume surge
            if (price > upper_w_aligned[i] and 
                squeeze[i] and 
                vol_surge[i]):
                signals[i] = 0.25
                position = 1
                continue
            
            # Short: price breaks below weekly Keltner lower band during Bollinger squeeze + volume surge
            elif (price < lower_w_aligned[i] and 
                  squeeze[i] and 
                  vol_surge[i]):
                signals[i] = -0.25
                position = -1
                continue
        
        elif position == 1:
            # Exit long: price returns to weekly Keltner middle or squeeze ends
            if (price < ema20_w_aligned[i] or 
                not squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to weekly Keltner middle or squeeze ends
            if (price > ema20_w_aligned[i] or 
                not squeeze[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals