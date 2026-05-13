#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (>1.8x 20-bar avg volume).
# Uses Camarilla pivot levels from daily timeframe for structure, 1d EMA34 for smooth trend alignment, and high volume threshold to reduce false signals.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag while capturing strong momentum moves.
# Exit on reverse Camarilla (R3/S3) touch or volume drop below 60% of average.

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
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels (R3, S3) from prior 1d bar
    lookback = 1
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    # Camarilla: R3 = close + 1.1*(high-low)*1.25/2, S3 = close - 1.1*(high-low)*1.25/2
    camarilla_upper = (close_series + 1.1 * (high_series - low_series) * 1.25 / 2).shift(lookback).values
    camarilla_lower = (close_series - 1.1 * (high_series - low_series) * 1.25 / 2).shift(lookback).values
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(20, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_upper[i]) or 
            np.isnan(camarilla_lower[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3, close > 1d EMA34, volume spike (>1.8x avg)
            if (high[i] > camarilla_upper[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Camarilla S3, close < 1d EMA34, volume spike (>1.8x avg)
            elif (low[i] < camarilla_lower[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if price breaks below Camarilla S3 or volume drops
            if (low[i] < camarilla_lower[i]) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if price breaks above Camarilla R3 or volume drops
            if (high[i] > camarilla_upper[i]) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals