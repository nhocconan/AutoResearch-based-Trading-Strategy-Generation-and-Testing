#!/usr/bin/env python3
"""
6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: On 6h timeframe, Camarilla R3/S3 breakouts aligned with 1d EMA34 trend and volume confirmation (>1.5x average) capture institutional moves with controlled frequency. Weekly trend filter (price vs weekly EMA50) adds regime alignment to avoid counter-trend whipsaws in both bull and bear markets. Discrete sizing (0.25) minimizes fee churn while targeting 50-150 total trades over 4 years.
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
    
    # Load 1d data ONCE before loop for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA50 for weekly trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d EMA34 for intraday trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar (R3/S3 for breakout, R4/S4 for strong breakout)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    camarilla_r4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    camarilla_s4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align to 1d (wait for completed 1d bar)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # ATR(14) for volatility
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Average volume for confirmation (24-period SMA = 1d on 6h timeframe)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    base_size = 0.25
    
    # Warmup: max of EMA(50), EMA(34), volume(24)
    start_idx = max(50, 34, 24)
    
    for i in range(start_idx, n):
        close_val = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_1w_val = ema_50_1w_aligned[i]
        ema_1d_val = ema_34_1d_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        s4_val = camarilla_s4_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(ema_1w_val) or np.isnan(ema_1d_val) or np.isnan(avg_vol) or 
            np.isnan(r3_val) or np.isnan(s3_val) or np.isnan(r4_val) or np.isnan(s4_val)):
            # Hold current position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirmed = vol > 1.5 * avg_vol
        
        # Trend filters: price vs weekly EMA50 and 1d EMA34
        weekly_uptrend = close_val > ema_1w_val
        weekly_downtrend = close_val < ema_1w_val
        daily_uptrend = close_val > ema_1d_val
        daily_downtrend = close_val < ema_1d_val
        
        # Long: price CLOSES above R3 with weekly AND daily uptrend and volume
        long_condition = (close_val > r3_val) and weekly_uptrend and daily_uptrend and volume_confirmed
        # Strong long: price CLOSES above R4 with weekly AND daily uptrend and volume
        strong_long_condition = (close_val > r4_val) and weekly_uptrend and daily_uptrend and volume_confirmed
        
        # Short: price CLOSES below S3 with weekly AND daily downtrend and volume
        short_condition = (close_val < s3_val) and weekly_downtrend and daily_downtrend and volume_confirmed
        # Strong short: price CLOSES below S4 with weekly AND daily downtrend and volume
        strong_short_condition = (close_val < s4_val) and weekly_downtrend and daily_downtrend and volume_confirmed
        
        # Exit: price retests broken level (R3/S3 for normal, R4/S4 for strong)
        long_exit = (position == 1 and close_val <= r3_val)
        strong_long_exit = (position == 1 and close_val <= r4_val)
        short_exit = (position == -1 and close_val >= s3_val)
        strong_short_exit = (position == -1 and close_val >= s4_val)
        
        # Entry logic: prefer strong breakouts, fallback to normal
        if (strong_long_condition or long_condition) and position != 1:
            signals[i] = base_size
            position = 1
            entry_price = close_val
        elif (strong_short_condition or short_condition) and position != -1:
            signals[i] = -base_size
            position = -1
            entry_price = close_val
        elif (strong_long_exit or long_exit):
            signals[i] = 0.0
            position = 0
        elif (strong_short_exit or short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
    
    return signals

name = "6h_Camarilla_R3S3_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0