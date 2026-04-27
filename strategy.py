#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm
Hypothesis: Uses 4h Camarilla pivot points (R1/S1) for breakout entries on 1h timeframe.
Enter long when price breaks above R1 AND 4h close > EMA50 (uptrend) AND volume > 1.5 * 20-period average.
Enter short when price breaks below S1 AND 4h close < EMA50 (downtrend) AND volume > 1.5 * 20-period average.
Exit when price returns to the pivot point (PP) OR trend reverses.
Session filter: 08-20 UTC to avoid low-liquidity hours.
Position size: 0.20 (20% of capital) to manage drawdown in bear markets.
Target: 60-150 total trades over 4 years (15-37/year) to minimize fee drag.
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
    
    # Get 4h data for Camarilla pivots and trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate typical price for 4h
    typical_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    
    # 4h EMA50 for trend filter
    close_4h_series = pd.Series(df_4h['close'].values)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels on 4h data
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # R1 = C + Range * 1.1/12, S1 = C - Range * 1.1/12
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    typical_4h_val = (high_4h + low_4h + close_4h) / 3
    range_4h = high_4h - low_4h
    
    camarilla_pp = typical_4h_val
    camarilla_r1 = close_4h + range_4h * 1.1 / 12
    camarilla_s1 = close_4h - range_4h * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_4h, camarilla_pp)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is already datetime64[ms]
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need 4h EMA50 (50), volume avg (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(volume_confirm[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_50_4h_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla levels with 4h trend filter AND volume
            # Long: price breaks above R1 AND 4h uptrend AND volume
            long_condition = (close_val > r1_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S1 AND 4h downtrend AND volume
            short_condition = (close_val < s1_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot point OR trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot point OR trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeConfirm"
timeframe = "1h"
leverage = 1.0