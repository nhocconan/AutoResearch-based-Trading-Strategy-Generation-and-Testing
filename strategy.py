# 4h_Donchian20_1dTrend_VolumeBreakout
# Hypothesis: Uses daily trend filter with Donchian channel breakout and volume confirmation.
# Enters long when price breaks above 20-period Donchian high with 1d uptrend (price > 1d EMA50) and volume spike.
# Enters short when price breaks below 20-period Donchian low with 1d downtrend (price < 1d EMA50) and volume spike.
# Daily trend filter ensures alignment with intermediate trend; volume confirms breakout strength; Donchian breakouts capture momentum.
# Targets 20-40 trades/year on 4h timeframe.

name = "4h_Donchian20_1dTrend_VolumeBreakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema_50_1d_4h = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel on 4h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike filter on 4h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: Price > Donchian high, above 1d EMA50 trend, volume spike
            if close[i] > donchian_high[i] and close[i] > ema_50_1d_4h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: Price < Donchian low, below 1d EMA50 trend, volume spike
            elif close[i] < donchian_low[i] and close[i] < ema_50_1d_4h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        elif position == 1:
            # Exit: price breaks below Donchian low or below 1d EMA50
            if close[i] < donchian_low[i] or close[i] < ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or above 1d EMA50
            if close[i] > donchian_high[i] or close[i] > ema_50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
    
    return signals