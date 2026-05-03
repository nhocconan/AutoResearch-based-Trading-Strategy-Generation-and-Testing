#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Williams %R breakout with 1w EMA(34) trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; breakout from these levels with
# weekly trend alignment and volume spike captures strong moves in both bull/bear markets.
# Discrete position sizing (0.25) minimizes fee drag while maintaining edge.
# Target: 30-100 trades over 4 years (7-25/year) to avoid overtrading.

name = "1d_WilliamsR14_1wEMA34_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Williams %R(14) on 1d timeframe
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14)
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Williams %R levels: oversold < -80, overbought > -20
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Trend filter: price above/below 1w EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Williams %R breaks above oversold (-80) + above 1w EMA + volume spike
            if williams_r[i] > -80 and williams_r[i-1] <= -80 and price_above_ema and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R breaks below overbought (-20) + below 1w EMA + volume spike
            elif williams_r[i] < -20 and williams_r[i-1] >= -20 and price_below_ema and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R reaches overbought (-20) or loses trend alignment
            if williams_r[i] >= -20 or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R reaches oversold (-80) or loses trend alignment
            if williams_r[i] <= -80 or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals