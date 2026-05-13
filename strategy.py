#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume spike confirmation.
# Long when price breaks above Camarilla R3 and close > 1d EMA34 with volume > 2.0x 20-bar average.
# Short when price breaks below Camarilla S3 and close < 1d EMA34 with volume > 2.0x 20-bar average.
# Uses discrete sizing 0.25 to target 50-150 total trades over 4 years on 12h timeframe.
# Camarilla levels from 1d provide structure; 1d EMA34 ensures trend alignment; volume confirms momentum.
# Works in bull via breakouts and in bear via mean-reversion at extremes. Designed for low trade frequency.

name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from 1d (using previous day's OHLC)
    # We need previous day's high, low, close
    df_1d_prev = df_1d.copy()
    df_1d_prev['high_prev'] = df_1d['high'].shift(1)
    df_1d_prev['low_prev'] = df_1d['low'].shift(1)
    df_1d_prev['close_prev'] = df_1d['close'].shift(1)
    
    # Calculate Camarilla R3 and S3 levels
    # R3 = close + (high - low) * 1.1 / 4
    # S3 = close - (high - low) * 1.1 / 4
    high_prev = df_1d_prev['high_prev'].values
    low_prev = df_1d_prev['low_prev'].values
    close_prev = df_1d_prev['close_prev'].values
    
    camarilla_r3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    camarilla_s3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_prev, camarilla_s3)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike
            if (high[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 2.0 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike
            elif (low[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 2.0 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S3 OR volume drops below average
            if (low[i] < camarilla_s3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R3 OR volume drops below average
            if (high[i] > camarilla_r3_aligned[i] or 
                volume[i] < avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals