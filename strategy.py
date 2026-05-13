#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation (>1.5x 20-bar avg volume).
# Williams %R identifies overbought/oversold conditions; 1d EMA34 provides trend bias; volume spike confirms momentum.
# Designed for low trade frequency (target 50-150 total over 4 years) to minimize fee drag while capturing reversals in ranging markets.
# Exit on reverse Williams %R cross or volume drop below 50% of average.

name = "6h_WilliamsR_MeanReversion_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    
    # Calculate Williams %R (14-period) based on prior 14 bars
    lookback = 14
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using shift(1) to ensure we only use prior bars (no look-ahead)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    highest_high = high_series.rolling(window=lookback, min_periods=lookback).max().shift(1).values
    lowest_low = low_series.rolling(window=lookback, min_periods=lookback).min().shift(1).values
    williams_r = (highest_high - close_series) / (highest_high - lowest_low) * -100
    williams_r = williams_r.shift(1).values  # Align with prior bar calculation
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback, 20), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price > 1d EMA34, volume spike (>1.5x avg)
            if (williams_r[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.5 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price < 1d EMA34, volume spike (>1.5x avg)
            elif (williams_r[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.5 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if Williams %R crosses above -50 or volume drops
            if (williams_r[i] > -50) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if Williams %R crosses below -50 or volume drops
            if (williams_r[i] < -50) or (volume[i] < 0.5 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals