#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with volume confirmation (1.5x MA20) and ADX trend filter (ADX>25).
# Enters long when price breaks above Donchian upper channel with ADX>25 and volume>1.5x MA20.
# Enters short when price breaks below Donchian lower channel with ADX>25 and volume>1.5x MA20.
# Exits when price reverts to the Donchian midpoint or ATR-based stoploss (2.0 * ATR(14) from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence: price breakout + ADX trend + volume spike.
# Donchian channels provide objective breakout levels, while ADX filter ensures trending markets only.
# Volume confirmation reduces false breakouts, improving signal quality in both bull and bear markets.

name = "4h_Donchian20_Breakout_ADX_Trend_Volume_v1"
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # ADX(14) for trend filter
    plus_dm = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    minus_dm = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    plus_di = 100 * (pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr)
    dx = 100 * np.absolute(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # ATR(14) for stoploss
    atr14 = atr  # already calculated above
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after sufficient data for all indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(adx[i]) or np.isnan(vol_ma20[i]) or np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with ADX>25 and volume spike
            if close[i] > highest_high[i] and adx[i] > 25 and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
            # SHORT: Price breaks below Donchian lower channel with ADX>25 and volume spike
            elif close[i] < lowest_low[i] and adx[i] > 25 and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Donchian midpoint OR ATR stoploss hit
            if close[i] < donchian_mid[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # EXIT SHORT: Price reverts to Donchian midpoint OR ATR stoploss hit
            if close[i] > donchian_mid[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals