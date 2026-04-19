#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily 1-week pivot point (S1/R1) breakout with volume confirmation and ATR-based stop.
# Weekly pivot levels act as strong support/resistance in both bull and bear markets.
# Breakouts with volume indicate institutional interest. Works in bull/bear as
# mean-reversion at pivot levels captures reversals, while breakouts catch trends.
# Target: 15-30 trades over 4 years to minimize fee drag.

name = "1d_1wPivot_S1R1_Breakout_VolumeATR_Simple"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points and ATR
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly ATR(14)
    tr1 = np.maximum(high_1w[1:], close_1w[:-1]) - np.minimum(low_1w[1:], close_1w[:-1])
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1w = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Weekly pivot points: P = (H+L+C)/3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    s1_1w = 2 * pivot_1w - high_1w
    r1_1w = 2 * pivot_1w - low_1w
    
    # Align to daily timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or np.isnan(atr_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        atr = atr_1w_aligned[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        s1 = s1_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume
            if price > r1 and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume
            elif price < s1 and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price closes below S1 or ATR stop
            if price < s1 or price < (high[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price closes above R1 or ATR stop
            if price > r1 or price > (low[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals