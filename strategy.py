#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla L3/H3 breakout with 1-day volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla H3 level with volume > 2x 20-period average and chop < 61.8 (trending)
# Short when price breaks below 1d Camarilla L3 level with volume > 2x 20-period average and chop < 61.8 (trending)
# Exit when price returns to 1d Camarilla Pivot level
# Uses 1-day Camarilla levels for institutional reference points, volume for confirmation, and chop to avoid ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing meaningful moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, H3, H2, H1, L1, L2, L3, L4
    # H3 = Close + 1.1 * (High - Low) / 6
    # L3 = Close - 1.1 * (High - Low) / 6
    # Pivot = (High + Low + Close) / 3
    camarilla_h3 = close_1d + 1.1 * (high_1d - low_1d) / 6
    camarilla_l3 = close_1d - 1.1 * (high_1d - low_1d) / 6
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Choppiness Index on 1d (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(period)
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr1 = np.concatenate([[0], tr1])  # First TR is 0
    atr_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum()
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max()
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min
    chop = 100 * np.log10(atr_sum / (hh - ll)) / np.log10(14)
    chop = chop.values
    
    # Align indicators to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pivot)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 30  # for 20-period calculations and chop
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_pivot_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_1d_current = volume[i]  # Current volume (12h bar approximates 1d volume for spike detection)
        
        if position == 0:
            # Long setup: break above Camarilla H3 with volume spike and trending market (chop < 61.8)
            if (price > camarilla_h3_aligned[i] and 
                vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                chop_aligned[i] < 61.8):                        # Trending market
                position = 1
                signals[i] = position_size
            # Short setup: break below Camarilla L3 with volume spike and trending market
            elif (price < camarilla_l3_aligned[i] and 
                  vol_1d_current > 2.0 * vol_ma_1d_aligned[i] and  # Volume spike
                  chop_aligned[i] < 61.8):                        # Trending market
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Camarilla Pivot
            if price > camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Camarilla Pivot
            if price < camarilla_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_L3H3_Volume_Chop"
timeframe = "12h"
leverage = 1.0