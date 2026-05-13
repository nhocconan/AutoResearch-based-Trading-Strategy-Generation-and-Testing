#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation (1.5x MA20).
# Enters long when price breaks above Donchian upper channel with 1d bullish trend (close > EMA50) and volume > 1.5x MA20.
# Enters short when price breaks below Donchian lower channel with 1d bearish trend (close < EMA50) and volume > 1.5x MA20.
# Exits when price reverts to Donchian middle (20-period mean) or ATR-based stoploss hit (2.0 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~19-50/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Donchian channels provide structural breakout signals, while 1d EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.5x) reduces false breakouts, improving signal quality in both bull and bear markets.
# Works in bull markets by catching breakouts and in bear markets by shorting breakdowns with trend alignment.

name = "4h_Donchian20_Breakout_1dEMA50_Volume_v1"
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high + lowest_low) / 2.0
    
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
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(donchian_middle[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with 1d bullish trend and volume spike
            if close[i] > highest_high[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Donchian lower channel with 1d bearish trend and volume spike
            elif close[i] < lowest_low[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Donchian middle OR ATR stoploss hit
            if close[i] < donchian_middle[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Donchian middle OR ATR stoploss hit
            if close[i] > donchian_middle[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals