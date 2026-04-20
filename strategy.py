# [67368] Hypothesis: 12h timeframe with 1w/1d HTF, using 1w high/low breakout + volume confirmation + 1d trend filter.  
# Why: 1w trend captures long-term direction, breakout captures momentum, volume confirms, 1d EMA avoids counter-trend trades.  
# Works in bull (breakouts up) and bear (breakouts down). Target: 15-30 trades/year.  
# Avoids overtrading by requiring 1w breakout (rare) + volume spike + trend alignment.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_HighLowBreakout_VolumeTrend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === Weekly High/Low (for breakout) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Previous week's values (to avoid look-ahead)
    prev_high_1w = np.roll(high_1w, 1)
    prev_low_1w = np.roll(low_1w, 1)
    prev_high_1w[0] = high_1w[0]
    prev_low_1w[0] = low_1w[0]
    
    # Align to 12h timeframe
    high_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_high_1w)
    low_1w_aligned = align_htf_to_ltf(prices, df_1w, prev_low_1w)
    
    # === Daily EMA for trend filter (50 > 200 = uptrend, < = downtrend) ===
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # === Volume confirmation (12h volume > 2x 20-period average) ===
    volume = prices['volume'].values
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        high_1w_val = high_1w_aligned[i]
        low_1w_val = low_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        ema50_val = ema50_1d_aligned[i]
        ema200_val = ema200_1d_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(high_1w_val) or 
            np.isnan(low_1w_val) or np.isnan(vol_ratio_val) or 
            np.isnan(ema50_val) or np.isnan(ema200_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above weekly high with volume confirmation and uptrend
            if close_val > high_1w_val and vol_ratio_val > 2.0 and ema50_val > ema200_val:
                signals[i] = 0.25
                position = 1
            # Short: Break below weekly low with volume confirmation and downtrend
            elif close_val < low_1w_val and vol_ratio_val > 2.0 and ema50_val < ema200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below weekly low OR trend breaks down
            if close_val < low_1w_val or ema50_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above weekly high OR trend breaks up
            if close_val > high_1w_val or ema50_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals