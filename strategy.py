#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter, volume confirmation (1.8x MA20), and ATR stoploss (2.0 * ATR14).
# Enters long when price breaks above Camarilla R3 with 4h bullish trend (close > EMA50), volume > 1.8x MA20.
# Enters short when price breaks below Camarilla S3 with 4h bearish trend (close < EMA50), volume > 1.8x MA20.
# Exits when price reverts to Camarilla pivot point or ATR-based stoploss hit (2.0 * ATR14 from entry).
# Uses discrete position sizing (0.20) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~15-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels provide precise intraday support/resistance, while 4h EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.8x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume_v1"
timeframe = "1h"
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
    
    # Get 4h data for trend filter (EMA50) and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from 4h OHLC
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    pivot = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    r3 = pivot + (range_4h * 1.1 / 4)
    s3 = pivot - (range_4h * 1.1 / 4)
    r4 = pivot + (range_4h * 1.1 / 2)
    s4 = pivot - (range_4h * 1.1 / 2)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_4h, pivot)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
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
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 4h bullish trend and volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 4h bearish trend and volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot point OR ATR stoploss hit
            if close[i] < pivot_aligned[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.20
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot point OR ATR stoploss hit
            if close[i] > pivot_aligned[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.20
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals