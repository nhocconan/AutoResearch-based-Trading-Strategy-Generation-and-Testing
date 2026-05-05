#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout + 4h EMA50 trend filter + volume confirmation
# Long when: price breaks above Camarilla R3 AND 4h EMA50 rising AND volume > 1.5x 20-period MA
# Short when: price breaks below Camarilla S3 AND 4h EMA50 falling AND volume > 1.5x 20-period MA
# Exit when: price returns to Camarilla pivot point OR trend reverses
# Uses Camarilla for structure, 4h EMA for trend filter, volume for conviction
# Timeframe: 1h, HTF: 4h. Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeConfirm"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    # Calculate volume confirmation on 1h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 4h data ONCE before loop for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 50-period EMA on 4h timeframe
    if len(close_4h) >= 50:
        ema_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_slope = np.diff(ema_4h, prepend=np.nan)
        ema_rising = ema_slope > 0
        ema_falling = ema_slope < 0
    else:
        ema_rising = np.full(len(close_4h), False)
        ema_falling = np.full(len(close_4h), False)
    
    # Align 4h EMA trend to 1h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_4h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_4h, ema_falling.astype(float))
    
    # Calculate Camarilla levels on 1h (using previous bar's OHLC)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), 
    #            S3 = close - 1.1*(high-low), S4 = close - 1.5*(high-low)
    # Pivot = (high + low + close) / 3
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        # Shift by 1 to use previous bar's OHLC (no look-ahead)
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        camarilla_pivot = (prev_high + prev_low + prev_close) / 3
        camarilla_range = prev_high - prev_low
        camarilla_r3 = prev_close + 1.1 * camarilla_range
        camarilla_s3 = prev_close - 1.1 * camarilla_range
    else:
        camarilla_pivot = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Apply session filter
        if not session_filter[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + 4h EMA50 rising + volume filter
            if (close[i] > camarilla_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + 4h EMA50 falling + volume filter
            elif (close[i] < camarilla_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot OR 4h EMA turns falling
            if (close[i] <= camarilla_pivot[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla pivot OR 4h EMA turns rising
            if (close[i] >= camarilla_pivot[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals