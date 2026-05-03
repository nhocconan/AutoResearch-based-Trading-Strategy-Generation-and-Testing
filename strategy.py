#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
# Designed to capture major trend continuations aligned with weekly structure while filtering noise.
# Uses discrete position sizing (0.30) to balance profit potential and drawdown control.
# Works in bull/bear markets by following 1w EMA34 direction and requiring volume confirmation for validity.
# Target: 30-100 trades over 4 years (7-25/year) to minimize fee drag while maintaining edge.

name = "1d_Donchian20_1wEMA34_VolumeSpike"
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
    
    # Volume confirmation: 20-period EMA on 1d volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d Donchian(20) channels from completed 1d bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start from 50 to have valid Donchian and volume EMA
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 1.8 x 20-period EMA (stricter to reduce trades)
        volume_spike = volume[i] > (1.8 * vol_ema_20[i])
        
        # Trend filter: price above/below 1w EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian(20) high + above 1w EMA34 + volume spike
            if close[i] > donchian_high[i] and price_above_ema and volume_spike:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian(20) low + below 1w EMA34 + volume spike
            elif close[i] < donchian_low[i] and price_below_ema and volume_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian(20) low or loses 1w trend alignment
            if close[i] < donchian_low[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above Donchian(20) high or loses 1w trend alignment
            if close[i] > donchian_high[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals