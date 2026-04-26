#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeRegime
Hypothesis: Camarilla R1/S1 breakouts (weaker levels = more frequent signals) with 12h EMA50 trend filter and volume confirmation. Uses 12h EMA > price for long bias, EMA < price for short bias to align with higher timeframe trend. Volume ensures participation. Fixed size 0.25 targets 20-30 trades/year. Works in bull/bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous 12h bar's OHLC for Camarilla levels (R1/S1 = standard breakout levels)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_vals = df_12h['close'].values
    
    # Calculate Camarilla levels: R1, S1 (standard breakout levels)
    rng = high_12h - low_12h
    camarilla_r1 = close_12h_vals + (rng * 1.1 / 12)   # R1 level
    camarilla_s1 = close_12h_vals - (rng * 1.1 / 12)   # S1 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume regime: volume > 60th percentile of 30-period lookback
    vol_series = pd.Series(volume)
    vol_percentile_60 = vol_series.rolling(window=30, min_periods=30).quantile(0.60).values
    volume_regime = volume > vol_percentile_60
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA, 30 for volume percentile)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(vol_percentile_60[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r1_val = camarilla_r1_aligned[i]
        camarilla_s1_val = camarilla_s1_aligned[i]
        ema_12h_val = ema_50_12h_aligned[i]
        vol_regime = volume_regime[i]
        size = fixed_size
        
        # Trend filter: 12h EMA50 > price = uptrend bias, < price = downtrend bias
        uptrend_bias = ema_12h_val < close_val   # price above EMA = uptrend
        downtrend_bias = ema_12h_val > close_val  # price below EMA = downtrend
        
        # Entry conditions: breakout of Camarilla R1/S1 with volume regime AND trend bias
        long_entry = (close_val > camarilla_r1_val) and vol_regime and uptrend_bias
        short_entry = (close_val < camarilla_s1_val) and vol_regime and downtrend_bias
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val < mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on mean reversion to midpoint (Camarilla center)
            mid_point = (camarilla_r1_val + camarilla_s1_val) / 2
            if close_val > mid_point:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Trend_VolumeRegime"
timeframe = "4h"
leverage = 1.0