#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Daily trend filter (price vs daily EMA50) + Camarilla R1/S1 breakouts from 12h with volume spike (>2x average) captures strong institutional moves while avoiding counter-trend whipsaws. Designed for 12h to target 12-37 trades/year with discrete sizing (0.25).
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
    
    # Load 1d data ONCE before loop for trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 12h bar (need 12h data)
    # Since we're on 12h timeframe, we use previous 12h bar for Camarilla calculation
    # We'll resample internally for Camarilla but use actual 12h bars from prices
    # For true 12h Camarilla, we need to calculate from the 12h OHLC itself
    # But prices is already 12h, so we can use its own high/low/close shifted by 1
    high_12h = high
    low_12h = low
    close_12h = close
    
    # Previous 12h bar's OHLC for Camarilla calculation
    high_12h_prev = np.roll(high_12h, 1)
    low_12h_prev = np.roll(low_12h, 1)
    close_12h_prev = np.roll(close_12h, 1)
    high_12h_prev[0] = 0
    low_12h_prev[0] = 0
    close_12h_prev[0] = 0
    
    camarilla_r1 = close_12h_prev + (high_12h_prev - low_12h_prev) * 1.1 / 12
    camarilla_s1 = close_12h_prev - (high_12h_prev - low_12h_prev) * 1.1 / 12
    
    # Align to 12h (wait for completed 12h bar) - already aligned since using same timeframe
    camarilla_r1_aligned = camarilla_r1  # No alignment needed for same TF
    camarilla_s1_aligned = camarilla_s1  # No alignment needed for same TF
    
    # ATR(14) for volatility (used in volume spike threshold)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 2 * 12h = 1d)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(50), volume(24)
    start_idx = max(50, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_val = ema_50_1d_aligned[i]
        r1_val = camarilla_r1_aligned[i]
        s1_val = camarilla_s1_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_val) or np.isnan(avg_vol) or np.isnan(r1_val) or 
            np.isnan(s1_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirmed = vol > 2.0 * avg_vol
        
        # Trend filter: price vs daily EMA50
        uptrend = close_val > ema_val
        downtrend = close_val < ema_val
        
        # Long: price CLOSES above R1 with daily uptrend and volume
        long_condition = (close_val > r1_val) and uptrend and volume_confirmed
        # Short: price CLOSES below S1 with daily downtrend and volume
        short_condition = (close_val < s1_val) and downtrend and volume_confirmed
        
        # Exit: price retests broken level
        long_exit = (position == 1 and close_val <= r1_val)
        short_exit = (position == -1 and close_val >= s1_val)
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0