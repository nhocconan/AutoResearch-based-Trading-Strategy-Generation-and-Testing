#!/usr/bin/env python3
# 6h_donchian_weekly_pivot_volume_v2
# Hypothesis: 6h strategy using Donchian channel breakout (20) aligned with weekly pivot direction and volume confirmation.
# Enters long when price breaks above Donchian upper (20) AND weekly pivot bias is bullish AND volume spike (>1.5x 20-period avg).
# Enters short when price breaks below Donchian lower (20) AND weekly pivot bias is bearish AND volume spike.
# Uses discrete sizing (±0.25) to balance return and drawdown. Target: 50-150 total trades over 4 years.
# Weekly pivot bias determines market regime: bullish if weekly close > weekly open, bearish otherwise.
# This avoids whipsaw in ranging markets by requiring alignment with higher timeframe momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_weekly_pivot_volume_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot bias: bullish if weekly close > weekly open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open  # True for bullish weekly candle
    
    # Align weekly bias to 6h timeframe (completed weekly candle only)
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Donchian channel (20-period) on 6h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period volume average on 6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)  # Volume at least 1.5x average
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price falls below Donchian lower (20) or weekly bias turns bearish
            if close[i] < lowest_20[i] or weekly_bullish_aligned[i] < 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price rises above Donchian upper (20) or weekly bias turns bullish
            if close[i] > highest_20[i] or weekly_bullish_aligned[i] >= 0.5:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price breaks above Donchian upper (20) AND weekly bias bullish AND volume spike
            if (close[i] > highest_20[i]) and (weekly_bullish_aligned[i] >= 0.5) and vol_spike[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: price breaks below Donchian lower (20) AND weekly bias bearish AND volume spike
            elif (close[i] < lowest_20[i]) and (weekly_bullish_aligned[i] < 0.5) and vol_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals