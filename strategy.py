#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter, volume confirmation (2.0x MA20), and ATR stoploss (1.5 * ATR14).
# Enters long when price breaks above Camarilla R3 with 1w bullish trend (close > EMA34), volume > 2.0x MA20.
# Enters short when price breaks below Camarilla S3 with 1w bearish trend (close < EMA34), volume > 2.0x MA20.
# Exits when price reverts to Camarilla pivot point or ATR-based stoploss hit (1.5 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~7-25/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels provide high-probability reversal points, while 1w EMA34 filter ensures alignment with higher timeframe momentum.
# Volume threshold (2.0x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Get 1w data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA(34) on 1w close
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Camarilla levels from 1w data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = np.roll(close_1w, 1)
    close_1w_shift[0] = close_1w[0]  # first bar
    camarilla_pivot = (high_1w + low_1w + close_1w_shift) / 3.0
    camarilla_range = high_1w - low_1w
    camarilla_r3 = camarilla_pivot + (camarilla_range * 1.1 / 4)
    camarilla_s3 = camarilla_pivot - (camarilla_range * 1.1 / 4)
    camarilla_r4 = camarilla_pivot + (camarilla_range * 1.1 / 2)
    camarilla_s4 = camarilla_pivot - (camarilla_range * 1.1 / 2)
    
    # Align Camarilla levels to 1d timeframe
    camarilla_pivot_aligned = align_htf_to_ltf(prices, df_1w, camarilla_pivot)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s4)
    
    # Volume filter: current volume > 2.0x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)
    
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
        if np.isnan(camarilla_pivot_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or \
           np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1w bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1w bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot point (mean reversion) OR ATR stoploss hit
            if close[i] < camarilla_pivot_aligned[i] or close[i] < entry_price[i-1] - 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot point (mean reversion) OR ATR stoploss hit
            if close[i] > camarilla_pivot_aligned[i] or close[i] > entry_price[i-1] + 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals