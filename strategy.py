#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses weekly Camarilla pivot levels (R3/S3) for breakout entries on daily timeframe.
Enter long when price breaks above weekly R3 AND 1w close > EMA34 (uptrend) AND volume > 2.0 * 20-period average.
Enter short when price breaks below weekly S3 AND 1w close < EMA34 (downtrend) AND volume > 2.0 * 20-period average.
Exit when price returns to weekly pivot (PP) level OR trend reverses.
Weekly trend filter ensures alignment with higher timeframe structure. High volume threshold (2.0x) filters weak breakouts.
Designed to work in both bull and bear markets via trend filter and breakout logic. Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1w data for Camarilla pivots and weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla pivots on 1w data (using previous week's OHLC)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Shift by 1 to get previous week's OHLC for this week's Camarilla levels
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 4.0)
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 4.0)
    
    # Align 1w Camarilla levels to 1d timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pp)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1w EMA34 (34), volume avg (20), 1w data shifted (1)
    start_idx = max(34, 20, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_1w_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 levels with 1w trend filter AND volume
            # Long: price breaks above R3 AND 1w uptrend AND volume
            long_condition = (close_val > r3_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S3 AND 1w downtrend AND volume
            short_condition = (close_val < s3_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot level OR trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot level OR trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0