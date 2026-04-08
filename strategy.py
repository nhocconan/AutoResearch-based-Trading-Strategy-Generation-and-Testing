#!/usr/bin/env python3
# 1d_1w_camarilla_pivot_volume_v1
# Hypothesis: Daily Camarilla pivot levels with weekly trend filter and volume confirmation. Captures institutional reversal levels while avoiding countertrend trades. Works in bull/bear via mean reversion at extremes with trend alignment.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_pivot_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation (use previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    # Align weekly EMA to daily (no additional delay needed for EMA)
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: H4 = C + (H-L)*1.1/2, L4 = C - (H-L)*1.1/2
    #          H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    #          H2 = C + (H-L)*1.1/6, L2 = C - (H-L)*1.1/6
    #          H1 = C + (H-L)*1.1/12, L1 = C - (H-L)*1.1/12
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate levels
    H4 = prev_close + (prev_high - prev_low) * 1.1 / 2
    L4 = prev_close - (prev_high - prev_low) * 1.1 / 2
    H3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    L3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    H2 = prev_close + (prev_high - prev_low) * 1.1 / 6
    L2 = prev_close - (prev_high - prev_low) * 1.1 / 6
    H1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    L1 = prev_close - (prev_high - prev_low) * 1.1 / 12
    
    # Align Camarilla levels to daily (shifted by 1 day already)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H2_aligned = align_htf_to_ltf(prices, df_1d, H2)
    L2_aligned = align_htf_to_ltf(prices, df_1d, L2)
    H1_aligned = align_htf_to_ltf(prices, df_1d, H1)
    L1_aligned = align_htf_to_ltf(prices, df_1d, L1)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below H3 (take profit at resistance)
            if close[i] < H3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: price closes above L3 (take profit at support)
            if close[i] > L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: price touches L4 (strong support) with volume and above weekly EMA200 (uptrend)
            if close[i] <= L4_aligned[i] and vol_confirm[i] and close[i] > ema200_1w_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H4 (strong resistance) with volume and below weekly EMA200 (downtrend)
            elif close[i] >= H4_aligned[i] and vol_confirm[i] and close[i] < ema200_1w_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals