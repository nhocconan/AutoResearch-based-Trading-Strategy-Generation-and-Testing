#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (1.5x MA20) and ATR-based trailing stop (2.0 * ATR14).
# Enters long when price breaks above Donchian upper channel with volume > 1.5x MA20.
# Enters short when price breaks below Donchian lower channel with volume > 1.5x MA20.
# Exits long when price crosses below Donchian middle (20-period mean) or ATR stoploss hit (2.0 * ATR14 from highest high since entry).
# Exits short when price crosses above Donchian middle or ATR stoploss hit (2.0 * ATR14 from lowest low since entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring volume confirmation on breakouts.
# Donchian channels provide clear trend structure, volume filter reduces false breakouts.
# ATR stoploss adapts to volatility, improving performance in both bull and bear markets.

name = "4h_Donchian20_Breakout_Volume_ATRStop_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_middle = close_series.rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # ATR(14) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track position and entry extremes for trailing stop
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high = np.full(n, np.nan)  # highest high since long entry
    lowest_low = np.full(n, np.nan)    # lowest low since short entry
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or \
           np.isnan(donchian_middle[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with volume spike
            if close[i] > donchian_upper[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                highest_high[i] = high[i]  # initialize tracking
            # SHORT: Price breaks below Donchian lower with volume spike
            elif close[i] < donchian_lower[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                lowest_low[i] = low[i]   # initialize tracking
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking arrays
                if i > 0:
                    highest_high[i] = highest_high[i-1]
                    lowest_low[i] = lowest_low[i-1]
        elif position == 1:
            # Update highest high since entry
            highest_high[i] = max(highest_high[i-1], high[i])
            # EXIT LONG: Price crosses below Donchian middle OR ATR stoploss hit
            if close[i] < donchian_middle[i] or close[i] < highest_high[i] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                # Reset tracking arrays
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                signals[i] = 0.25
        elif position == -1:
            # Update lowest low since entry
            lowest_low[i] = min(lowest_low[i-1], low[i])
            # EXIT SHORT: Price crosses above Donchian middle OR ATR stoploss hit
            if close[i] > donchian_middle[i] or close[i] > lowest_low[i] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                # Reset tracking arrays
                highest_high[i] = np.nan
                lowest_low[i] = np.nan
            else:
                signals[i] = -0.25
    
    return signals