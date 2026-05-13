#!/usr/bin/env python3
# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume spike confirmation.
# Long when price breaks above 4h Camarilla R3 level with price > 1d EMA34 (bullish trend) and volume > 2.0x 20-bar average.
# Short when price breaks below 4h Camarilla S3 level with price < 1d EMA34 (bearish trend) and volume > 2.0x average.
# Exit when price reverses and closes below/above the 4h Camarilla pivot point (PP).
# Uses discrete position sizing 0.20. Target: 60-150 total trades over 4 years on 1h timeframe.
# Uses 4h/1d for signal direction, 1h only for entry timing. Session filter 08-20 UTC to reduce noise.
# EMA trend filter ensures we trade with higher timeframe trend, avoiding counter-trend whipsaws.
# Volume confirmation validates breakout strength. Camarilla PP exit provides clear, objective stop.

name = "1h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "1h"
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar: based on previous 4h bar's OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formulas: PP = (H+L+C)/3, Range = H-L
    # R3 = PP + (H-L) * 1.1/2, S3 = PP - (H-L) * 1.1/2
    pp_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3_4h = pp_4h + range_4h * 1.1 / 2.0
    s3_4h = pp_4h - range_4h * 1.1 / 2.0
    
    # Align Camarilla levels to 1h timeframe (wait for 4h bar to close)
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pp_4h_aligned = align_htf_to_ltf(prices, df_4h, pp_4h)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    if len(close_1d) < 34:
        ema_34_1d = np.full(len(close_1d), np.nan)
    else:
        ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA to 1h timeframe (wait for 1d bar to close)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for volume average
        # Skip if any required data is NaN or outside session
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(pp_4h_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(avg_volume[i]) or not in_session[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above R3 with bullish 1d EMA trend and volume spike
            if (close[i] > r3_4h_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S3 with bearish 1d EMA trend and volume spike
            elif (close[i] < s3_4h_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below pivot point (reversal signal)
            if close[i] < pp_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price closes above pivot point (reversal signal)
            if close[i] > pp_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals