#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R(14) with 1d EMA(34) trend filter and volume spike confirmation
# Williams %R identifies overbought/oversold conditions; combined with 1d EMA trend filter
# and volume confirmation to capture reversals aligned with daily momentum.
# Designed for 6h timeframe to reduce trade frequency and minimize fee drag.
# Works in bull/bear markets by following 1d EMA34 direction and requiring oversold/overbought
# conditions with volume spike for entry. Discrete position sizing (0.25) balances risk/return.

name = "6h_WilliamsR14_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Avoid division by zero
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: 20-period EMA on 6h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Trend filter: price above/below 1d EMA34
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Williams %R conditions: oversold (< -80) for long, overbought (> -20) for short
        williams_oversold = williams_r[i] < -80
        williams_overbought = williams_r[i] > -20
        
        if position == 0:
            # Long: Williams %R oversold + above 1d EMA + volume spike
            if williams_oversold and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought + below 1d EMA + volume spike
            elif williams_overbought and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or loses trend alignment
            if williams_r[i] > -50 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or loses trend alignment
            if williams_r[i] < -50 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals