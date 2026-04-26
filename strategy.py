#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_v1
Hypothesis: Trade 4h Camarilla R1/S1 breakouts with 12h EMA34 trend filter and volume confirmation.
R1/S1 levels provide frequent but reliable breakout opportunities when aligned with 12h trend.
Volume confirmation ensures breakouts have institutional participation.
ATR-based stoploss limits downside during false breakouts.
Targets 100-200 total trades over 4 years (25-50/year) to balance opportunity and fee drag.
Works in both bull and bear markets by following the 12h trend direction.
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
    
    # Get 12h data for trend filter and Camarilla calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 12h for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 12h bar
    prev_high = df_12h['high'].shift(1).values
    prev_low = df_12h['low'].shift(1).values
    prev_close = df_12h['close'].shift(1).values
    
    # Handle NaN from shift
    prev_high = np.where(np.isnan(prev_high), df_12h['high'].values, prev_high)
    prev_low = np.where(np.isnan(prev_low), df_12h['low'].values, prev_low)
    prev_close = np.where(np.isnan(prev_close), df_12h['close'].values, prev_close)
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 6.0)
    s1 = pivot - (range_hl * 1.1 / 6.0)
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    
    # Align Camarilla levels to 4h
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r3_aligned = align_htf_to_ltf(prices, df_12h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_12h, s3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average (~4d average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # ATR(14) for dynamic stoploss
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of 12h EMA(34), volume MA(20), ATR(14), and need 12h data
    start_idx = max(34, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(atr[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        close_val = close[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        trend_up = close_val > ema_34_12h_aligned[i]   # 12h uptrend
        trend_down = close_val < ema_34_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Long: price breaks above R1 AND volume confirm AND 12h uptrend
            long_signal = (close_val > r1_aligned[i]) and vol_conf and trend_up
            
            # Short: price breaks below S1 AND volume confirm AND 12h downtrend
            short_signal = (close_val < s1_aligned[i]) and vol_conf and trend_down
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions:
            # 1. Stoploss: price drops below entry - 1.5 * ATR
            # 2. Take profit: price reaches R3 (strong resistance)
            # 3. Trend flip: 12h trend turns down
            if (close_val < entry_price - 1.5 * atr_val) or \
               (close_val > r3_aligned[i]) or \
               (not trend_up):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions:
            # 1. Stoploss: price rises above entry + 1.5 * ATR
            # 2. Take profit: price reaches S3 (strong support)
            # 3. Trend flip: 12h trend turns up
            if (close_val > entry_price + 1.5 * atr_val) or \
               (close_val < s3_aligned[i]) or \
               (not trend_down):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_12hTrend_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0