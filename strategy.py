#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(50) trend filter and volume spike confirmation
# Designed to capture medium-term breakouts aligned with daily trend while filtering false moves.
# Uses discrete position sizing (0.30) to balance profit potential and drawdown control.
# Works in bull/bear markets by following 1d EMA50 direction and requiring volume confirmation for validity.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while maintaining edge.
# Timeframe: 12h, HTF: 1d

name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) from completed 12h bars
    # We need to calculate this on 12h data, so we'll use the prices directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (wait for completed 1d bar)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA on 12h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start from 60 to have valid Donchian and EMA
        # Skip if any value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (stricter to reduce trades)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + above 1d EMA50 + volume spike
            if close[i] > donchian_high[i] and price_above_ema and volume_spike:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below Donchian low + below 1d EMA50 + volume spike
            elif close[i] < donchian_low[i] and price_below_ema and volume_spike:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or loses 1d trend alignment
            if close[i] < donchian_low[i] or not price_above_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit short: price breaks above Donchian high or loses 1d trend alignment
            if close[i] > donchian_high[i] or not price_below_ema:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals