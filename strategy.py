#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Reversal_1dTrend_Filter_v3
Hypothesis: 12h reversal at daily Camarilla R4/S4 levels (extreme institutional support/resistance) when price shows rejection (wick > body) and aligns with 1d EMA50 trend. Uses volume confirmation (>1.5x 20-bar MA) to filter false signals. Designed for low trade frequency (12-30 trades/year) to minimize fee drag in bear markets. Works in both bull and bear by fading extremes in trend direction - counter-trend at Camarilla extremes but only with trend alignment to avoid catching falling knives.
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
    open_ = prices['open'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter (slower = more reliable)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R4, S4 levels (extreme levels - 1.1/2 range from close)
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Wick ratio: (wick size) / (body size) > 1.5 indicates rejection
    body_size = np.abs(close - open_)
    upper_wick = high - np.maximum(close, open_)
    lower_wick = np.minimum(close, open_) - low
    # Avoid division by zero
    wick_ratio = np.where(body_size > 0, 
                         np.maximum(upper_wick, lower_wick) / body_size, 
                         0.0)
    strong_rejection = wick_ratio > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    base_size = 0.25  # Position size
    
    # Warmup: max of calculations (20 for vol, 50 for ema)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            bars_since_entry += 1 if position != 0 else 0
            continue
        
        close_val = close[i]
        ema_50_val = ema_50_1d_aligned[i]
        camarilla_r4_val = camarilla_r4_aligned[i]
        camarilla_s4_val = camarilla_s4_aligned[i]
        vol_spike = volume_spike[i]
        rejection = strong_rejection[i]
        
        # Determine 1d trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1d = close_val > ema_50_val
        bearish_1d = close_val < ema_50_val
        
        # Entry conditions: rejection at Camarilla extreme in trend direction with volume
        # Long: rejection at S4 (support) in bullish trend
        # Short: rejection at R4 (resistance) in bearish trend
        long_entry = (close_val <= camarilla_s4_val * 1.005) and rejection and bullish_1d and vol_spike
        short_entry = (close_val >= camarilla_r4_val * 0.995) and rejection and bearish_1d and vol_spike
        
        # Exit conditions: opposite Camarilla level or trend reversal
        exit_long = (close_val >= camarilla_r4_val) or not bullish_1d
        exit_short = (close_val <= camarilla_s4_val) or not bearish_1d
        
        # Minimum holding period: 2 bars to avoid whipsaw
        min_hold = 2
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -base_size
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_long:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = base_size
                bars_since_entry += 1
        elif position == -1:
            # Short - check exit conditions only after minimum hold
            if bars_since_entry >= min_hold and exit_short:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -base_size
                bars_since_entry += 1
    
    return signals

name = "12h_Camarilla_Pivot_Reversal_1dTrend_Filter_v3"
timeframe = "12h"
leverage = 1.0