#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime
Hypothesis: Uses 12h Camarilla R3/S3 breakouts filtered by 1d EMA34 trend and volume spike. 
Enters long when price breaks above R3, 1d close > EMA34 (uptrend), and volume > 2x average.
Enters short when price breaks below S3, 1d close < EMA34 (downtrend), and volume > 2x average.
Exits when price returns to 12h pivot point (PP) or 1d trend reverses.
Designed for 12h timeframe with very tight entries to minimize fee drag: target 12-25 trades/year.
Works in both bull and bear markets via 1d trend filter - only trades with the higher timeframe trend.
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
    
    # Get 12h data for Camarilla pivots and entry timing
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Camarilla pivots on 12h data (using previous 12h bar's OHLC)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Shift by 1 to get previous 12h bar's OHLC for current Camarilla levels
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_close = np.roll(close_12h, 1)
    # First value will be invalid (rolled from last), set to nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla levels (R3/S3 are stronger breakout levels)
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 2.0)  # R3 level
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 2.0)  # S3 level
    
    # Align 12h Camarilla levels to 12h timeframe (identity alignment)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_12h, camarilla_pp)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 12h data shifted (1), 1d EMA34 (34), volume avg (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_pp_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R3/S3 levels with 1d trend filter and volume
            # Long: price breaks above R3 AND 1d uptrend AND volume
            long_condition = (close_val > r3_level) and (close_val > ema_1d_val) and vol_conf
            # Short: price breaks below S3 AND 1d downtrend AND volume
            short_condition = (close_val < s3_level) and (close_val < ema_1d_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot level OR 1d trend breaks
            exit_condition = (close_val <= pp_level) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot level OR 1d trend breaks
            exit_condition = (close_val >= pp_level) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_Regime"
timeframe = "12h"
leverage = 1.0