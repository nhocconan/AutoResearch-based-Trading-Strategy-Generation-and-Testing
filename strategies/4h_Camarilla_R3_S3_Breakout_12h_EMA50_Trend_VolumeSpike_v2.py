#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_12h_EMA50_Trend_VolumeSpike_v2
Hypothesis: Trade Camarilla R3/S3 breakouts on 4h with 12h EMA50 trend filter and volume spike confirmation. Uses ATR trailing stop. 12h EMA50 provides stronger trend filter than 1d EMA34 for 4h timeframe, reducing false breakouts in choppy markets. Works in both bull (long at R3 breakout above EMA50) and bear (short at S3 breakdown below EMA50). Target: 75-200 total trades over 4 years = 19-50/year.
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
    
    # Get 12h data for EMA50 trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Camarilla levels from previous 12h bar (HLC of 12h)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_arr = df_12h['close'].values
    
    # Calculate Camarilla R3 and S3 for each 12h bar
    camarilla_r3_12h = close_12h_arr + ((high_12h - low_12h) * 1.1 / 4)
    camarilla_s3_12h = close_12h_arr - ((high_12h - low_12h) * 1.1 / 4)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3_12h)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3_12h)
    
    # ATR for stoploss and volatility filter
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    # Warmup: max of EMA50 (50), ATR (14), volume MA (20)
    start_idx = max(50, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_val = ema_50_12h_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        r3_level = camarilla_r3_aligned[i]
        s3_level = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: Break above R3, above 12h EMA50, with volume spike
            long_signal = (high_val > r3_level) and (close_val > ema_50_val) and vol_spike
            
            # Short: Break below S3, below 12h EMA50, with volume spike
            short_signal = (low_val < s3_level) and (close_val < ema_50_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                highest_since_entry = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                lowest_since_entry = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            highest_since_entry = max(highest_since_entry, close_val)
            # Exit: Close below EMA50 (trend change) OR trailing stop (2.5*ATR below high)
            if (close_val < ema_50_val) or (close_val < highest_since_entry - 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            lowest_since_entry = min(lowest_since_entry, close_val)
            # Exit: Close above EMA50 (trend change) OR trailing stop (2.5*ATR above low)
            if (close_val > ema_50_val) or (close_val > lowest_since_entry + 2.5 * atr_val):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_12h_EMA50_Trend_VolumeSpike_v2"
timeframe = "4h"
leverage = 1.0