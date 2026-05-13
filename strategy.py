#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation (1.8x MA20), and ATR-based trailing stop (2.0 * ATR14).
# Enters long when price breaks above Donchian upper band with 1d bullish trend (close > EMA50), volume > 1.8x MA20.
# Enters short when price breaks below Donchian lower band with 1d bearish trend (close < EMA50), volume > 1.8x MA20.
# Exits when price reverts to Donchian midpoint or trailing stoploss hit (highest high since entry - 2.0 * ATR14 for longs, lowest low since entry + 2.0 * ATR14 for shorts).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Donchian channels provide clear trend-following structure, while 1d EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.8x) reduces false breakouts, improving signal quality in both bull and bear markets.
# Trailing stop allows profits to run while controlling risk.

name = "12h_Donchian20_Breakout_1dTrend_Volume_TS_v1"
timeframe = "12h"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian channels (20-period) from 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    # ATR(14) for trailing stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Track entry price and extreme prices for trailing stop
    entry_price = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)  # for long positions
    lowest_low = np.full(n, np.nan)    # for short positions
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(donchian_mid_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high with 1d bullish trend and volume spike
            if close[i] > donchian_high_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]
                highest_high[i] = close[i]
            # SHORT: Price breaks below Donchian low with 1d bearish trend and volume spike
            elif close[i] < donchian_low_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]
                lowest_low[i] = close[i]
            else:
                signals[i] = 0.0
                # Carry forward NaN values for tracking arrays
                if i > 0:
                    entry_price[i] = entry_price[i-1]
                    highest_high[i] = highest_high[i-1]
                    lowest_low[i] = lowest_low[i-1]
        elif position == 1:
            # Update highest high for trailing stop
            highest_high[i] = max(highest_high[i-1], high[i])
            # EXIT LONG: Price reverts to Donchian midpoint OR trailing stoploss hit
            trailing_stop = highest_high[i] - 2.0 * atr14[i]
            if close[i] < donchian_mid_aligned[i] or close[i] < trailing_stop:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
                highest_high[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]
        elif position == -1:
            # Update lowest low for trailing stop
            lowest_low[i] = min(lowest_low[i-1], low[i])
            # EXIT SHORT: Price reverts to Donchian midpoint OR trailing stoploss hit
            trailing_stop = lowest_low[i] + 2.0 * atr14[i]
            if close[i] > donchian_mid_aligned[i] or close[i] > trailing_stop:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
                lowest_low[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]
    
    return signals