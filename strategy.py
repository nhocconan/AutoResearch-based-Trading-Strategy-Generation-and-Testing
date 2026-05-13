#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for trend alignment, 12h Camarilla R3/S3 for breakout entry, and volume spike (>1.8x 20-bar avg) for confirmation.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag on 12h timeframe.
# Works in both bull and bear markets by following the 1d trend direction and requiring volume confirmation to avoid false breakouts.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter (HTF) - called ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 12h bar (primary TF)
    lookback = 1
    # Camarilla R3, S3 based on prior bar's high, low, close
    prior_high = high[-lookback] if len(high) > lookback else high[-1]
    prior_low = low[-lookback] if len(low) > lookback else low[-1]
    prior_close = close[-lookback] if len(close) > lookback else close[-1]
    R3 = ((prior_high - prior_low) * 1.1 / 4) + prior_close
    S3 = prior_close - ((prior_high - prior_low) * 1.1 / 4)
    # Shift to align with current bar
    R3_series = pd.Series([R3] * n).values
    S3_series = pd.Series([S3] * n).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after volume lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R3_series[i]) or 
            np.isnan(S3_series[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike (>1.8x avg)
            if (high[i] > R3_series[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike (>1.8x avg)
            elif (low[i] < S3_series[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S3 or volume drops
            if (low[i] < S3_series[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R3 or volume drops
            if (high[i] > R3_series[i]) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals