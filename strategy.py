#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 level and close > 1w EMA34 with volume > 1.8x 20-bar average.
# Short when price breaks below Camarilla S3 level and close < 1w EMA34 with volume > 1.8x 20-bar average.
# Uses discrete sizing 0.25 to target 30-100 total trades over 4 years on 1d timeframe.
# Camarilla levels provide tighter structure than Donchian for mean reversion in ranging markets,
# while EMA34 filter ensures we only trade with the weekly trend. Volume spike confirms conviction.
# Designed to work in both bull (trend continuation) and bear (mean reversion off weekly trend) markets.

name = "1d_Camarilla_R3S3_Breakout_1wEMA34_Trend_VolumeConfirm"
timeframe = "1d"
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
    
    lookback = 20  # for Camarilla calculation and volume average
    
    # Calculate Camarilla levels (based on previous day's range)
    # Camarilla R3 = close_prev + 1.1 * (high_prev - low_prev) / 2
    # Camarilla S3 = close_prev - 1.1 * (high_prev - low_prev) / 2
    close_prev = pd.Series(close).shift(1).values
    high_prev = pd.Series(high).shift(1).values
    low_prev = pd.Series(low).shift(1).values
    
    camarilla_r3 = close_prev + 1.1 * (high_prev - low_prev) / 2
    camarilla_s3 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(34) on 1w close
    if len(close_1w) < 34:
        ema_34_1w = np.full(len(close_1w), np.nan)
    else:
        ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA to 1d timeframe (wait for 1w bar to close)
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=lookback, min_periods=lookback).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1w EMA34, volume spike
            if (high[i] > camarilla_r3[i] and 
                close[i] > ema_34_1w_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1w EMA34, volume spike
            elif (low[i] < camarilla_s3[i] and 
                  close[i] < ema_34_1w_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume dries up (< 0.8x average)
            if (low[i] < camarilla_s3[i] or 
                volume[i] < 0.8 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume dries up (< 0.8x average)
            if (high[i] > camarilla_r3[i] or 
                volume[i] < 0.8 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals