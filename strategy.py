#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout + 1w EMA34 trend + volume confirmation
# Long when: price breaks above R3, 1w close > EMA34, volume > 1.5x 24-period MA (12h equivalent)
# Short when: price breaks below S3, 1w close < EMA34, volume > 1.5x 24-period MA
# Exit when: price retouches the 1d pivot point (PP) or volume drops below average
# Uses Camarilla levels for precise entry, 1w EMA for major trend filter, volume for conviction
# Timeframe: 12h, HTF: 1w. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_Camarilla_R3S3_Breakout_1wEMA34_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate volume confirmation on 12h using 24-period MA (equivalent to 1d lookback)
    if len(volume) >= 24:
        vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
        volume_filter = volume > (1.5 * vol_ma_24)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1w data ONCE before loop for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 34-period EMA for 1w trend filter
    if len(close_1w) >= 34:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    else:
        ema_34_1w = np.full(len(close_1w), np.nan)
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Get 1d data ONCE before loop for Camarilla levels calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    camarilla_pp = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i == 0:  # Skip first bar as we need previous day
            continue
        # Use previous day's OHLC to calculate today's Camarilla levels
        phigh = high_1d[i-1]
        plow = low_1d[i-1]
        pclose = close_1d[i-1]
        
        # Camarilla calculations
        range_val = phigh - plow
        camarilla_pp[i] = (phigh + plow + pclose) / 3
        camarilla_r3[i] = camarilla_pp[i] + range_val * 1.1 / 4
        camarilla_s3[i] = camarilla_pp[i] - range_val * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 1w bullish trend, volume confirmation
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and  # Price above 1w EMA34
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S3, 1w bearish trend, volume confirmation
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and  # Price below 1w EMA34
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retouches pivot point or volume drops
            if (close[i] <= pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches pivot point or volume drops
            if (close[i] >= pp_aligned[i] or not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals