#!/usr/bin/env python3
# Hypothesis: 6h Williams %R mean reversion with 1d EMA34 trend filter and volume confirmation (>1.6x 20-bar avg volume).
# Williams %R identifies overbought/oversold conditions. In ranging markets (BTC/ETH 2025), mean reversion at extremes works.
# Trend filter ensures trades align with higher timeframe momentum. Volume confirmation reduces false signals.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag. Position size: 0.25.

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
    
    # Calculate Williams %R (14-period) on 6h data
    lookback_willr = 14
    highest_high = pd.Series(high).rolling(window=lookback_willr, min_periods=lookback_willr).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_willr, min_periods=lookback_willr).min().values
    willr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    willr = np.where((highest_high - lowest_low) == 0, -50, willr)
    
    # Calculate average volume for confirmation (20-period)
    lookback_vol = 20
    avg_volume = pd.Series(volume).rolling(window=lookback_vol, min_periods=lookback_vol).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(max(lookback_willr, lookback_vol, 1), n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(willr[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80), price above 1d EMA34, volume spike
            if (willr[i] < -80 and 
                close[i] > ema_34_1d_aligned[i] and 
                volume[i] > 1.6 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20), price below 1d EMA34, volume spike
            elif (willr[i] > -20 and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume[i] > 1.6 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to neutral (> -50) or volume drops
            if (willr[i] > -50) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to neutral (< -50) or volume drops
            if (willr[i] < -50) or (volume[i] < 0.6 * avg_volume[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals