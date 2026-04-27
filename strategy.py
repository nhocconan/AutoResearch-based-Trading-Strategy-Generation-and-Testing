#!/usr/bin/env python3
"""
6h_IcebergDetector_LiquiditySweep_Reversal
Hypothesis: On 6h timeframe, price sweeps liquidity (recent swing high/low) then reverses 
when volume delta shows absorption (iceberg orders). Uses 1w trend filter and 1d volume 
profile to identify institutional footprints. Works in bull via long reversals at swept 
lows and in bear via short reversals at swept highs. Targets 60-100 trades over 4 years 
(15-25/year) with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume profile and swing points
    df_1d = get_htf_data(prices, '1d')
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d indicators: volume profile and swing points ===
    # Volume-weighted average price (VWAP) approximation for 1d
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d = vwap_1d.replace([np.inf, -np.inf], np.nan).ffill().bfill().values
    
    # Swing high/low (3-bar lookback/forward)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    swing_high = np.full(len(high_1d), np.nan)
    swing_low = np.full(len(low_1d), np.nan)
    
    for i in range(2, len(high_1d)-2):
        if high_1d[i] == np.max(high_1d[i-2:i+3]):
            swing_high[i] = high_1d[i]
        if low_1d[i] == np.min(low_1d[i-2:i+3]):
            swing_low[i] = low_1d[i]
    
    # Forward fill swing points for alignment
    swing_high_series = pd.Series(swing_high)
    swing_low_series = pd.Series(swing_low)
    swing_high_ffill = swing_high_series.ffill().bfill().values
    swing_low_ffill = swing_low_series.ffill().bfill().values
    
    # === 1w indicators: trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 6h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    swing_high_1d_aligned = align_htf_to_ltf(prices, df_1d, swing_high_ffill)
    swing_low_1d_aligned = align_htf_to_ltf(prices, df_1d, swing_low_ffill)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume delta (buying vs selling pressure approximation)
    # Using close position in range as proxy for volume delta
    close_pos = (close - low) / (high - low + 1e-10)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24*6h = 6d
    volume_ratio = volume / (vol_avg + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50_1w (50), volume avg (24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(swing_high_1d_aligned[i]) or 
            np.isnan(swing_low_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vwap_val = vwap_1d_aligned[i]
        swing_high_val = swing_high_1d_aligned[i]
        swing_low_val = swing_low_1d_aligned[i]
        ema_1w_val = ema_50_1w_aligned[i]
        vol_ratio = volume_ratio[i]
        close_pos_val = close_pos[i]
        
        # Determine 1w trend: price > EMA50 = uptrend, price < EMA50 = downtrend
        is_uptrend = close_val > ema_1w_val
        is_downtrend = close_val < ema_1w_val
        
        if position == 0:
            # Liquidity sweep detection with reversal signals
            # Bullish: sweep low (stop hunt) then close above VWAP with volume absorption
            bull_sweep = low_val < swing_low_val and close_val > vwap_val
            bull_absorption = close_pos_val > 0.6 and vol_ratio > 1.2  # Strong close + above avg volume
            
            # Bearish: sweep high (stop hunt) then close below VWAP with volume absorption
            bear_sweep = high_val > swing_high_val and close_val < vwap_val
            bear_absorption = close_pos_val < 0.4 and vol_ratio > 1.2  # Weak close + above avg volume
            
            if is_uptrend and bull_sweep and bull_absorption:
                signals[i] = size
                position = 1
            elif is_downtrend and bear_sweep and bear_absorption:
                signals[i] = -size
                position = -1
                
        elif position == 1:
            # Exit long: price reaches swing high or trend changes
            exit_condition = (close_val >= swing_high_val) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
                
        elif position == -1:
            # Exit short: price reaches swing low or trend changes
            exit_condition = (close_val <= swing_low_val) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_IcebergDetector_LiquiditySweep_Reversal"
timeframe = "6h"
leverage = 1.0