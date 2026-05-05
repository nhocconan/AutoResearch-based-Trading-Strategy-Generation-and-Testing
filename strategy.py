#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R3/S3 breakout with 12h EMA50 trend filter and volume spike confirmation
# Long when: price breaks above Camarilla R3 AND 12h EMA50 rising AND volume > 2x 20-period MA
# Short when: price breaks below Camarilla S3 AND 12h EMA50 falling AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla pivot point (PP) OR trend reverses
# Uses Camarilla pivot levels for structure, 12h EMA for trend, volume spike for conviction
# Timeframe: 4h, HTF: 12h. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla pivot levels on 4h using previous bar's OHLC
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    if len(high) >= 2 and len(low) >= 2 and len(close) >= 2:
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        pp = (prev_high + prev_low + prev_close) / 3.0
        camarilla_r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
        camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    else:
        pp = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
    
    # Get 12h data ONCE before loop for EMA calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 50-period EMA on 12h timeframe
    if len(close_12h) >= 50:
        ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_rising = np.diff(ema_50_12h, prepend=np.nan) > 0
        ema_falling = np.diff(ema_50_12h, prepend=np.nan) < 0
    else:
        ema_rising = np.full(len(close_12h), False)
        ema_falling = np.full(len(close_12h), False)
    
    # Align 12h EMA trend to 4h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(close[i]) or np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(pp[i]) or np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + 12h EMA rising + volume filter
            if (close[i] > camarilla_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S3 + 12h EMA falling + volume filter
            elif (close[i] < camarilla_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla pivot point OR 12h EMA turns falling
            if (close[i] <= pp[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla pivot point OR 12h EMA turns rising
            if (close[i] >= pp[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals