#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation.
Long when price breaks above 1h Camarilla R3 level AND price > 4h EMA34 (uptrend) AND volume > 1.8x average.
Short when price breaks below 1h Camarilla S3 level AND price < 4h EMA34 (downtrend) AND volume > 1.8x average.
Exit when price reverts to 1h Camarilla pivot point (PP) or trend reverses (price crosses 4h EMA34).
Uses 1h timeframe for entry timing with 4h trend filter to reduce noise and capture medium-term moves.
Volume confirmation ensures high-conviction breakouts. Session filter (08-20 UTC) reduces off-hours noise.
Target: 60-120 trades over 4 years (15-30/year) to stay within proven working range and avoid fee drag.
Works in bull markets via breakout momentum and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1h Camarilla levels (R3, S3, PP) - ONCE before loop
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 10:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Camarilla levels on 1h (based on previous 1h bar's OHLC)
    prev_high_1h = np.roll(high_1h, 1)
    prev_low_1h = np.roll(low_1h, 1)
    prev_close_1h = np.roll(close_1h, 1)
    # Set first value to NaN (no previous bar)
    prev_high_1h[0] = np.nan
    prev_low_1h[0] = np.nan
    prev_close_1h[0] = np.nan
    
    camarilla_pp = (prev_high_1h + prev_low_1h + prev_close_1h) / 3.0
    camarilla_r3 = prev_close_1h + (prev_high_1h - prev_low_1h) * 1.1 / 4.0  # R3
    camarilla_s3 = prev_close_1h - (prev_high_1h - prev_low_1h) * 1.1 / 4.0  # S3
    
    # Load 4h data for EMA34 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1h, camarilla_pp)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Volume average (20-period) on primary timeframe
    vol_ma_primary = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready or outside session
        if (np.isnan(camarilla_pp_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema34_4h_aligned[i]) or np.isnan(vol_ma_primary[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = camarilla_pp_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        ema34_val = ema34_4h_aligned[i]
        vol_ma_val = vol_ma_primary[i]
        
        # Get current price and volume
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above 1h Camarilla R3 AND price > 4h EMA34 (uptrend) AND volume confirmation
            if (price > r3_val and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below 1h Camarilla S3 AND price < 4h EMA34 (downtrend) AND volume confirmation
            elif (price < s3_val and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Camarilla PP OR price breaks below 4h EMA34 (trend reversal)
                if price <= pp_val or price < ema34_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Camarilla PP OR price breaks above 4h EMA34 (trend reversal)
                if price >= pp_val or price > ema34_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_Breakout_4hEMA34_Volume_Session"
timeframe = "1h"
leverage = 1.0