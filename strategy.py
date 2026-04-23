#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above R3 and 4h EMA50 > prior 4h EMA50 (uptrend) with volume > 1.5x average.
Short when price breaks below S3 and 4h EMA50 < prior 4h EMA50 (downtrend) with volume > 1.5x average.
Exit on opposite Camarilla level break or EMA50 direction change.
Camarilla levels provide precise support/resistance from prior day's range.
4h EMA50 filters for medium-term trend to avoid false breakouts in chop.
Volume confirmation ensures breakout legitimacy.
Designed for 1h timeframe targeting 60-150 total trades over 4 years with session filter (08-20 UTC) to reduce noise.
Works in both bull and bear markets by only taking breakouts in direction of 4h trend.
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
    
    # Load 4h data for EMA50 trend filter - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate EMA50 on 4h data
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_rising = np.zeros(len(close_4h), dtype=bool)
    ema_4h_falling = np.zeros(len(close_4h), dtype=bool)
    for i in range(1, len(ema_4h)):
        ema_4h_rising[i] = ema_4h[i] > ema_4h[i-1]
        ema_4h_falling[i] = ema_4h[i] < ema_4h[i-1]
    
    # Align 4h EMA50 trend to 1h timeframe
    ema_4h_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_rising)
    ema_4h_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_4h_falling)
    
    # Calculate Camarilla levels from prior 1d bar
    # Load 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    def calculate_camarilla(high, low, close):
        # Typical price for pivot
        pivot = (high + low + close) / 3
        range_val = high - low
        
        # Camarilla levels
        r3 = pivot + (range_val * 1.1 / 4)
        s3 = pivot - (range_val * 1.1 / 4)
        r4 = pivot + (range_val * 1.1 / 2)
        s4 = pivot - (range_val * 1.1 / 2)
        
        return r3, s3, r4, s4
    
    # Calculate Camarilla levels for each 1d bar (using prior bar's data)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        r3, s3, _, _ = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        camarilla_r3[i] = r3
        camarilla_s3[i] = s3
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(ema_4h_rising_aligned[i]) or np.isnan(ema_4h_falling_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_up = ema_4h_rising_aligned[i]
        ema_down = ema_4h_falling_aligned[i]
        r3_val = camarilla_r3_aligned[i]
        s3_val = camarilla_s3_aligned[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above R3 AND 4h EMA50 rising (uptrend) AND volume spike AND in session
            if (price > r3_val and ema_up and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.20
                position = 1
                entry_price = price
            # Short: price breaks below S3 AND 4h EMA50 falling (downtrend) AND volume spike AND in session
            elif (price < s3_val and ema_down and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.20
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S3 OR 4h EMA50 starts falling (trend weakening)
                if (price < s3_val or ema_down):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above R3 OR 4h EMA50 starts rising (trend weakening)
                if (price > r3_val or ema_up):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1H_Camarilla_R3_S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0