#!/usr/bin/env python3
# 4H_Donchian_20_Breakout_1dTrend_Volume
# Hypothesis: Uses 4h timeframe with Donchian channel breakout (20-period) and 1d trend filter (EMA50).
# Enters long when price breaks above Donchian upper band in uptrend (close > EMA50) with volume > 1.5x 20-period average.
# Enters short when price breaks below Donchian lower band in downtrend (close < EMA50) with volume confirmation.
# Exits when price returns to opposite Donchian band or trend reverses.
# Uses daily EMA50 for trend to avoid whipsaws and works in both bull/bear markets.
# Targets 20-50 trades per year on 4h timeframe with position size 0.25 to minimize fee drag.

name = "4H_Donchian_20_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # Get 1d data for EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 1d EMA50
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper in uptrend with volume spike
            if (close[i] > donchian_upper[i] and 
                price_above_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower in downtrend with volume spike
            elif (close[i] < donchian_lower[i] and 
                  price_below_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian lower or trend reverses to downtrend
            if (close[i] < donchian_lower[i] or 
                price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian upper or trend reverses to uptrend
            if (close[i] > donchian_upper[i] or 
                price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals