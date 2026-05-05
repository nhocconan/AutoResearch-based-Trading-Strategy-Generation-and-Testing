#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R3/S3 breakout + 1w EMA34 trend filter + volume spike confirmation
# Long when: price breaks above Camarilla R3 AND price > 1w EMA34 (uptrend) AND volume > 2x 20-period MA
# Short when: price breaks below Camarilla S3 AND price < 1w EMA34 (downtrend) AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla pivot point (mean reversion) OR trend reverses
# Uses Camarilla levels for institutional structure, 1w EMA for major trend filter, volume spike for conviction
# Timeframe: 1d, HTF: 1w. Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 1d using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels on 1d
    if len(high) >= 1 and len(low) >= 1 and len(close) >= 1:
        # Previous day's OHLC for Camarilla calculation
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        pivot = (prev_high + prev_low + prev_close) / 3.0
        range_hl = prev_high - prev_low
        
        # Camarilla R3 and S3 levels
        r3 = pivot + (range_hl * 1.1 / 4.0)
        s3 = pivot - (range_hl * 1.1 / 4.0)
        pp = pivot  # pivot point for exit
    else:
        r3 = np.full(n, np.nan)
        s3 = np.full(n, np.nan)
        pp = np.full(n, np.nan)
    
    # Breakout signals
    breakout_above_r3 = (close > r3) & (np.roll(close, 1) <= np.roll(r3, 1))
    breakout_below_s3 = (close < s3) & (np.roll(close, 1) >= np.roll(s3, 1))
    return_to_pp = (np.roll(close, 1) > pp) & (close <= pp)  # for long exit
    return_to_pp_short = (np.roll(close, 1) < pp) & (close >= pp)  # for short exit
    
    # Get 1w data ONCE before loop for EMA calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 34-period EMA on 1w timeframe
    if len(close_1w) >= 34:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_rising = np.diff(ema_34_1w, prepend=np.nan) > 0
        ema_falling = np.diff(ema_34_1w, prepend=np.nan) < 0
    else:
        ema_rising = np.full(len(close_1w), False)
        ema_falling = np.full(len(close_1w), False)
    
    # Align 1w EMA trend to 1d timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(pp[i]) or 
            np.isnan(ema_rising_aligned[i]) or np.isnan(ema_falling_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R3 + 1w EMA rising + volume filter
            if (breakout_above_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below S3 + 1w EMA falling + volume filter
            elif (breakout_below_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot point OR 1w EMA turns falling
            if (return_to_pp[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot point OR 1w EMA turns rising
            if (return_to_pp_short[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals