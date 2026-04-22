#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data once (no look-ahead)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's HLC for pivot calculation (no look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_close_1w = np.roll(close_1w, 1)
    prev_high_1w[0] = np.nan
    prev_low_1w[0] = np.nan
    prev_close_1w[0] = np.nan
    
    # Weekly pivot levels (standard formula)
    pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
    r4_1w = prev_high_1w + 3 * (pp_1w - prev_low_1w)  # R4 = High + 3*(PP - Low)
    s4_1w = prev_low_1w - 3 * (prev_high_1w - pp_1w)  # S4 = Low - 3*(High - PP)
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike detection (20-period average)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all HTF data to 1d timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        ema50 = ema50_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly R4 with volume + above daily EMA50
            if price > r4 and vol > 2.0 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly S4 with volume + below daily EMA50
            elif price < s4 and vol > 2.0 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to weekly central pivot (mean reversion)
            # Calculate weekly PP for exit signal
            pp_1w = (prev_high_1w + prev_low_1w + prev_close_1w) / 3
            pp_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
            pp = pp_aligned[i]
            
            if position == 1 and price < pp:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price > pp:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_WeeklyPivot_R4_S4_Breakout_1dEMA50_Volume_Spike"
timeframe = "1d"
leverage = 1.0