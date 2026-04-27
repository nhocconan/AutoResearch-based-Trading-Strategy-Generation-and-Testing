#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTF_v3
Hypothesis: Refined 12h strategy using daily Camarilla R3/S3 breakouts with 1d EMA34 trend filter and volume confirmation (>2.0x 20-period average). 
Enhanced with ATR-based trailing stop (2.5x ATR) and discrete position sizing (0.25) to reduce churn. 
Designed for BTC/ETH/SOL in both bull/bear markets via trend alignment and volume-filtered breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
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
    
    # Get 1d data for Camarilla pivots and daily trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate ATR for trailing stop (using 12h data)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Camarilla pivots on 1d data (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    camarilla_pp = (prev_high + prev_low + prev_close) / 3.0
    camarilla_range = prev_high - prev_low
    camarilla_r3 = camarilla_pp + (camarilla_range * 1.1 / 4.0)
    camarilla_s3 = camarilla_pp - (camarilla_range * 1.1 / 4.0)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    start_idx = max(34, 20, 14, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_pp_aligned[i]) or 
            np.isnan(volume_confirm[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        close_val = close[i]
        ema_val = ema_34_1d_aligned[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        pp_level = camarilla_pp_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            long_condition = (close_val > r3_level) and (close_val > ema_val) and vol_conf
            short_condition = (close_val < s3_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_condition:
                signals[i] = -size
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
        elif position == 1:
            highest_since_entry = max(highest_since_entry, close_val)
            exit_price = highest_since_entry - 2.5 * atr_val
            exit_condition = (close_val <= pp_level) or (close_val < ema_val) or (close_val <= exit_price)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, close_val)
            exit_price = lowest_since_entry + 2.5 * atr_val
            exit_condition = (close_val >= pp_level) or (close_val > ema_val) or (close_val >= exit_price)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_VolumeSpike_HTF_v3"
timeframe = "12h"
leverage = 1.0