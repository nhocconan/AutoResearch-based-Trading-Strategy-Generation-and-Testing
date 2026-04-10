#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and chop regime filter
# - Long when price breaks above Camarilla H3 level from previous 1d AND volume > 1.3x 20-bar avg AND chop < 61.8
# - Short when price breaks below Camarilla L3 level from previous 1d AND volume > 1.3x 20-bar avg AND chop < 61.8
# - Exit when price returns to Camarilla Pivot level (mean reversion to equilibrium)
# - Uses 1d Camarilla levels for structure, volume confirmation for momentum, chop filter to avoid ranging markets
# - Discrete position sizing (0.25) to minimize fee churn
# - Target: 25-35 trades/year on 4h timeframe (100-140 total over 4 years)
# - Camarilla pivots work well in 2025+ bear/range markets as price respects these levels

name = "4h_1d_camarilla_breakout_volume_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_p = np.zeros_like(close_1d)
    camarilla_h3 = np.zeros_like(close_1d)
    camarilla_l3 = np.zeros_like(close_1d)
    camarilla_h4 = np.zeros_like(close_1d)
    camarilla_l4 = np.zeros_like(close_1d)
    
    for i in range(len(df_1d)):
        high = high_1d[i]
        low = low_1d[i]
        close = close_1d[i]
        range_ = high - low
        
        camarilla_p[i] = (high + low + close * 2) / 4
        camarilla_h3[i] = camarilla_p[i] + range_ * 1.1 / 4
        camarilla_l3[i] = camarilla_p[i] - range_ * 1.1 / 4
        camarilla_h4[i] = camarilla_p[i] + range_ * 1.1 / 2
        camarilla_l4[i] = camarilla_p[i] - range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    p_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Pre-compute volume confirmation: > 1.3x 20-period average
    volume_20_avg = prices['volume'].rolling(window=20, min_periods=20).mean().values
    vol_spike = prices['volume'] > (1.3 * volume_20_avg)
    
    # Pre-compute Choppiness Index (CHOP) for regime filter
    def calculate_chop(high, low, close, window=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period TR
        
        # Sum of True Range over window
        atr_sum = pd.Series(tr).rolling(window=window, min_periods=window).sum().values
        
        # Highest high and lowest low over window
        hh = pd.Series(high).rolling(window=window, min_periods=window).max().values
        ll = pd.Series(low).rolling(window=window, min_periods=window).min().values
        
        # Choppiness Index
        chop = np.zeros_like(close)
        for i in range(len(close)):
            if atr_sum[i] > 0 and hh[i] != ll[i]:
                chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(window)
            else:
                chop[i] = 50.0  # Neutral value when undefined
        return chop
    
    chop = calculate_chop(prices['high'].values, prices['low'].values, prices['close'].values, 14)
    chop_filter = chop < 61.8  # Trending market (CHOP < 61.8)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(p_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or 
            np.isnan(l3_1d_aligned[i]) or np.isnan(volume_20_avg[i]) or 
            np.isnan(chop[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new breakout entries
            # Long when price breaks above H3 with volume spike in trending market
            if (prices['close'].iloc[i] > h3_1d_aligned[i] and 
                vol_spike.iloc[i] and 
                chop_filter.iloc[i]):
                position = 1
                signals[i] = 0.25
            # Short when price breaks below L3 with volume spike in trending market
            elif (prices['close'].iloc[i] < l3_1d_aligned[i] and 
                  vol_spike.iloc[i] and 
                  chop_filter.iloc[i]):
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Pivot level (mean reversion)
            exit_signal = False
            if position == 1:  # Long position
                if prices['close'].iloc[i] <= p_1d_aligned[i]:
                    exit_signal = True
            elif position == -1:  # Short position
                if prices['close'].iloc[i] >= p_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals