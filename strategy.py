#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h trend filter (close > 4h EMA50 for longs, close < 4h EMA50 for shorts) and volume confirmation.
Uses 4h EMA50 for trend direction and 1h for precise entry timing. Adds 08-20 UTC session filter to avoid low-liquidity periods.
Target: 15-37 trades/year per symbol (60-150 total over 4 years) with discrete position sizing (0.20) to minimize fee drag.
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
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - avoid look-ahead
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1h data for Camarilla levels - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla levels from previous 1h bar
    prev_high = np.roll(high_1h, 1)
    prev_low = np.roll(low_1h, 1)
    prev_close = np.roll(close_1h, 1)
    
    # First bar has no previous data
    prev_high[0] = high_1h[0]
    prev_low[0] = low_1h[0]
    prev_close[0] = close_1h[0]
    
    range_1h = prev_high - prev_low
    camarilla_r3 = prev_close + 1.1 * range_1h / 4
    camarilla_s3 = prev_close - 1.1 * range_1h / 4
    camarilla_h3 = prev_close + 1.1 * range_1h / 2
    camarilla_l3 = prev_close - 1.1 * range_1h / 2
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (20-period) on 1h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1h indicators to 1h timeframe (same timeframe, no alignment needed but keep for consistency)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_l3)
    
    # Align 4h EMA50 to 1h timeframe
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Camarilla R3 AND close > 4h EMA50 AND volume spike
            if (price > camarilla_r3_aligned[i] and 
                close[i] > ema50_4h_aligned[i] and 
                volume[i] > 2.0 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S3 AND close < 4h EMA50 AND volume spike
            elif (price < camarilla_s3_aligned[i] and 
                  close[i] < ema50_4h_aligned[i] and 
                  volume[i] > 2.0 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below Camarilla H3
                if price < camarilla_h3_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above Camarilla L3
                if price > camarilla_l3_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0