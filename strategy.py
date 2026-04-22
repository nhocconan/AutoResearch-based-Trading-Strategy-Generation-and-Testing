#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for HTF analysis
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot levels using previous day's HLC (no look-ahead)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    prev_close_1d[0] = np.nan
    
    pp_1d = (prev_high_1d + prev_low_1d + prev_close_1d) / 3
    r1_1d = 2 * pp_1d - prev_low_1d  # R1 = 2*PP - Low
    s1_1d = 2 * pp_1d - prev_high_1d  # S1 = 2*PP - High
    r2_1d = pp_1d + (high_1d - low_1d)  # R2 = PP + (High - Low)
    s2_1d = pp_1d - (high_1d - low_1d)  # S2 = PP - (High - Low)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align to 4h timeframe (primary timeframe)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume spike filter (20-period average on 4h data)
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index filter (14-period) - use 1d data for regime detection
    high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    chop_num = np.log10(atr_14.sum()) - np.log10((high_14 - low_14).sum())
    chop_den = np.log10(14)
    chop = 100 * chop_num / chop_den if chop_den != 0 else 50
    chop_series = pd.Series(np.full_like(close_1d, chop)).fillna(50).values
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_series)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        pp = pp_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        r2 = r2_aligned[i]
        s2 = s2_aligned[i]
        ema50 = ema50_aligned[i]
        chop_val = chop_aligned[i]
        
        # Only trade in trending markets (Chop < 50 indicates trend)
        if chop_val >= 50:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R2 with volume + above EMA50
            if price > r2 and vol > 1.5 * vol_ma and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S2 with volume + below EMA50
            elif price < s2 and vol > 1.5 * vol_ma and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price crosses back through S1/R1 or opposite extreme
            if position == 1 and (price < s1 or price > r2):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price > r1 or price < s2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Pivot_R2_S2_Breakout_1dEMA50_Volume_Trend"
timeframe = "4h"
leverage = 1.0