#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume spike.
# Long when price breaks above Camarilla R3 with 1d close > 1d EMA34 (bullish trend) and volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 with 1d close < 1d EMA34 (bearish trend) and volume > 2.0x average.
# Exit when price reverses and closes below/above the Camarilla pivot point (mean reversion exit).
# Uses discrete position sizing 0.25. Target: 75-200 total trades over 4 years on 4h timeframe.
# Camarilla levels provide precise intraday support/resistance that work in both trending and ranging markets.
# Volume confirmation validates breakout strength. 1d EMA filter ensures alignment with higher timeframe trend.
# Pivot point exit provides clear, objective mean reversion stop.

name = "4h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_price = prices['open'].values
    
    lookback = 20  # for volume average
    
    # Get 1d data for EMA trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    if len(close_1d) < 34:
        ema_34_1d = np.full(len(close_1d), np.nan)
    else:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 4h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R4 = C + ((H-L)*1.1/2), R3 = C + ((H-L)*1.1/4), etc.
    # We use the previous completed 1d bar to avoid look-ahead
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_pivot = np.full(n, np.nan)  # (H+L+C)/3
    
    for i in range(len(df_1d)):
        # Get the timestamp of this 1d bar
        bar_time = df_1d.index[i]
        # Find where this 1d bar ends in our 4h data
        # 1d bar ends at bar_time + 1 day
        end_time = bar_time + pd.Timedelta(days=1)
        # Find the first 4h bar that opens at or after end_time
        mask = prices['open_time'] >= end_time
        if mask.any():
            start_idx = mask.argmax()
            # Calculate Camarilla levels from this 1d bar
            h = high_1d[i]
            l = low_1d[i]
            c = close_1d[i]
            camarilla_pivot_val = (h + l + c) / 3.0
            camarilla_r3_val = c + ((h - l) * 1.1 / 4)
            camarilla_s3_val = c - ((h - l) * 1.1 / 4)
            # Fill all 4h bars from start_idx onwards with these levels
            camarilla_pivot[start_idx:] = camarilla_pivot_val
            camarilla_r3[start_idx:] = camarilla_r3_val
            camarilla_s3[start_idx:] = camarilla_s3_val
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_r3[i]) or 
            np.isnan(camarilla_s3[i]) or np.isnan(camarilla_pivot[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with bullish 1d EMA trend and volume spike
            if (close[i] > camarilla_r3[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3 with bearish 1d EMA trend and volume spike
            elif (close[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below Camarilla pivot point (mean reversion)
            if close[i] < camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above Camarilla pivot point (mean reversion)
            if close[i] > camarilla_pivot[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals