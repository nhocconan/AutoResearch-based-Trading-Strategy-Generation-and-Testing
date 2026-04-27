#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike
Hypothesis: Uses daily Camarilla pivot breakouts (R3/S3) filtered by 1-week EMA50 trend and volume spike (>2x average).
Enter long when daily price breaks above R3, 1-week close > 1-week EMA50 (uptrend), and volume > 2x average.
Enter short when daily price breaks below S3, 1-week close < 1-week EMA50 (downtrend), and volume > 2x average.
Exit when price returns to daily pivot (PP) or 1-week trend reverses.
Designed for 1d timeframe with tight entries to avoid fee drag: target 7-25 trades/year.
Works in both bull and bear markets via 1w trend filter and volume confirmation to avoid false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and entry timing
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivots on 1d data (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to get previous day's OHLC for current Camarilla levels
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 2.0)  # R3 = PP + 1.1*range/2
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 2.0)  # S3 = PP - 1.1*range/2
    
    # Align 1d Camarilla levels to 1d timeframe (identity alignment)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA50 for trend filter
    close_1w_series = pd.Series(df_1w['close'].values)
    ema_50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d data shifted (1), 1w EMA50 (50), volume avg (20)
    start_idx = max(1, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1w_val = ema_50_1w_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 levels with 1w trend filter and volume
            # Long: price breaks above R3 AND 1w uptrend AND volume
            long_condition = (close_val > r3_level) and (close_val > ema_1w_val) and vol_conf
            # Short: price breaks below S3 AND 1w downtrend AND volume
            short_condition = (close_val < s3_level) and (close_val < ema_1w_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot level OR 1w trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot level OR 1w trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_1w_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0