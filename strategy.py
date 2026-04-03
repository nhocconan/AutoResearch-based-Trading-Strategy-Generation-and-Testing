#!/usr/bin/env python3
"""
Experiment #1907: 6h Camarilla Pivot Breakout + 1d Trend Filter + Volume Confirmation
HYPOTHESIS: Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) from 1d timeframe provide institutional support/resistance. 
Strategy: 
- Use 1d Camarilla levels to determine bias: price > (R3+R4)/2 = bullish bias, price < (S3+S4)/2 = bearish bias
- Enter on 6h breakout of R4 (long) or S4 (short) only when aligned with 1d bias and volume > 1.5x 20-period average
- Exit when price returns to 1d VWAP (mean reversion) or opposite Camarilla level (R3/S3) is touched
- Works in bull/bear markets by following 1d institutional flow. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1907_6h_camarilla_1d_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels and trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + Range * 1.1/2
    # R3 = C + Range * 1.1/4
    # S3 = C - Range * 1.1/4
    # S4 = C - Range * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # 1d bias levels: midpoint between R3/R4 and S3/S4
    bull_bias_level = (r3_1d + r4_1d) / 2.0  # Above this = bullish bias
    bear_bias_level = (s3_1d + s4_1d) / 2.0  # Below this = bearish bias
    
    # Align 1d levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    bull_bias_aligned = align_htf_to_ltf(prices, df_1d, bull_bias_level)
    bear_bias_aligned = align_htf_to_ltf(prices, df_1d, bear_bias_level)
    
    # 1d trend filter: EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for EMA(50) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price returns to 1d VWAP approximation (mean reversion)
                # Using pivot as VWAP proxy
                if price <= pivot_1d_aligned[i]:
                    exit_signal = True
                # Exit if price touches S3 (strong support)
                elif price <= s3_1d_aligned[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to 1d VWAP approximation
                if price >= pivot_1d_aligned[i]:
                    exit_signal = True
                # Exit if price touches R3 (strong resistance)
                elif price >= r3_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above R4 AND 1d trend up AND price above bull bias level
            if trend_bias > 0 and price > r4_1d_aligned[i] and price > bull_bias_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below S4 AND 1d trend down AND price below bear bias level
            elif trend_bias < 0 and price < s4_1d_aligned[i] and price < bear_bias_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals