#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation
# Long when: price breaks above Camarilla R3 AND price > 1d EMA34 (uptrend) AND volume > 2x 20-period MA
# Short when: price breaks below Camarilla S3 AND price < 1d EMA34 (downtrend) AND volume > 2x 20-period MA
# Exit when: price returns to Camarilla pivot point (mean reversion) OR trend reverses
# Uses Camarilla levels for institutional support/resistance, 1d EMA for trend filter, volume spike for conviction
# Timeframe: 4h, HTF: 1d. Target: 75-200 total trades over 4 years (19-50/year) to avoid fee drag.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
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
    open_price = prices['open'].values  # Camarilla uses previous day's OHLC
    
    # Calculate volume confirmation on 4h using 20-period MA
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (2.0 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Calculate Camarilla levels for 4h using previous day's OHLC
    # Camarilla levels: H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    # R3 = C + 1.1*(H-L)/2, S3 = C - 1.1*(H-L)/2
    # Pivot = (H+L+C)/3
    # We need previous day's OHLC, so we shift by 1
    if len(high) >= 1 and len(low) >= 1 and len(close) >= 1 and len(open_price) >= 1:
        # Previous day's high, low, close
        prev_high = np.roll(high, 1)
        prev_low = np.roll(low, 1)
        prev_close = np.roll(close, 1)
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        # Calculate Camarilla levels
        H_minus_L = prev_high - prev_low
        camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
        camarilla_r3 = prev_close + 1.1 * H_minus_L / 2.0
        camarilla_s3 = prev_close - 1.1 * H_minus_L / 2.0
        
        # Breakout conditions
        breakout_above_r3 = close > camarilla_r3
        breakout_below_s3 = close < camarilla_s3
        return_to_pivot = np.abs(close - camarilla_pivot) < 0.001 * camarilla_pivot  # Within 0.1% of pivot
    else:
        camarilla_pivot = np.full(n, np.nan)
        camarilla_r3 = np.full(n, np.nan)
        camarilla_s3 = np.full(n, np.nan)
        breakout_above_r3 = np.zeros(n, dtype=bool)
        breakout_below_s3 = np.zeros(n, dtype=bool)
        return_to_pivot = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for EMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 34-period EMA on 1d timeframe
    if len(close_1d) >= 34:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
        ema_rising = np.diff(ema_34_1d, prepend=np.nan) > 0
        ema_falling = np.diff(ema_34_1d, prepend=np.nan) < 0
    else:
        ema_rising = np.full(len(close_1d), False)
        ema_falling = np.full(len(close_1d), False)
    
    # Align 1d EMA trend to 4h timeframe
    ema_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_rising.astype(float))
    ema_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_falling.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(ema_rising_aligned[i]) or 
            np.isnan(ema_falling_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: breakout above R3 + 1d EMA rising + volume filter
            if (breakout_above_r3[i] and 
                ema_rising_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: breakout below S3 + 1d EMA falling + volume filter
            elif (breakout_below_s3[i] and 
                  ema_falling_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to pivot OR 1d EMA turns falling
            if (return_to_pivot[i] or ema_falling_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to pivot OR 1d EMA turns rising
            if (return_to_pivot[i] or ema_rising_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals