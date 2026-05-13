#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (HMA21), volume confirmation (1.5x MA20), and ATR(14) stoploss.
# Enters long when price breaks above Donchian upper channel (20-period high) with 1w bullish trend (close > HMA21), volume > 1.5x MA20.
# Enters short when price breaks below Donchian lower channel (20-period low) with 1w bearish trend (close < HMA21), volume > 1.5x MA20.
# Exits via ATR-based stoploss (2 * ATR(14) from entry) or opposite Donchian breakout.
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~7-25/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Donchian channels provide clear structural breakouts, HMA reduces lag while smoothing noise, volume confirms conviction.
# The 1w trend filter ensures alignment with higher timeframe direction, avoiding counter-trend entries in choppy markets.

name = "1d_Donchian20_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def hma(values, period):
    """Hull Moving Average - reduces lag while maintaining smoothness"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = pd.Series(values).ewm(span=half, adjust=False, min_periods=half).mean().values
    wma1 = pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values
    raw_hma = 2 * wma2 - wma1
    hma_values = pd.Series(raw_hma).ewm(span=sqrt, adjust=False, min_periods=sqrt).mean().values
    return hma_values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter (HMA21)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate HMA(21) on 1w close
    hma21_1w = hma(close_1w, 21)
    hma21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma21_1w)
    
    # Donchian Channel (20) on 1d
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or \
           np.isnan(hma21_1w_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with 1w bullish trend and volume spike
            if close[i] > highest_20[i] and close[i] > hma21_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Donchian lower channel with 1w bearish trend and volume spike
            elif close[i] < lowest_20[i] and close[i] < hma21_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: ATR stoploss hit OR price breaks below Donchian lower channel (contrarian exit)
            if close[i] < entry_price[i-1] - 2.0 * atr14[i] or close[i] < lowest_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: ATR stoploss hit OR price breaks above Donchian upper channel (contrarian exit)
            if close[i] > entry_price[i-1] + 2.0 * atr14[i] or close[i] > highest_20[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals