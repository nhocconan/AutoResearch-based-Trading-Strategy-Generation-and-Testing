#!/usr/bin/env python3
# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (2.0x MA20).
# Enters long when price breaks above Camarilla R3 with 1d bullish trend (close > EMA34), volume > 2.0x MA20.
# Enters short when price breaks below Camarilla S3 with 1d bearish trend (close < EMA34), volume > 2.0x MA20.
# Exits when price reverts to Camarilla H3/L3 levels or ATR-based stoploss hit (1.5 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels from 1d provide institutional support/resistance, while 1d EMA34 filter ensures alignment with higher timeframe momentum.
# Volume threshold (2.0x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "6h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v1"
timeframe = "6h"
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # HLC from previous day: (high + low + close) / 3
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    # Camarilla levels based on previous day's range
    range_1d = high_1d - low_1d
    camarilla_h5 = typical_price + (range_1d * 1.1 / 2)
    camarilla_h4 = typical_price + (range_1d * 1.1 / 4)
    camarilla_h3 = typical_price + (range_1d * 1.1 / 6)
    camarilla_l3 = typical_price - (range_1d * 1.1 / 6)
    camarilla_l4 = typical_price - (range_1d * 1.1 / 4)
    camarilla_l5 = typical_price - (range_1d * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe (use previous day's levels)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
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
        if np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla H3 with 1d bullish trend and volume spike
            if close[i] > camarilla_h3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla L3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_l3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla H4 (mean reversion) OR ATR stoploss hit
            if close[i] < camarilla_h4_aligned[i] or close[i] < entry_price[i-1] - 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla L4 (mean reversion) OR ATR stoploss hit
            if close[i] > camarilla_l4_aligned[i] or close[i] > entry_price[i-1] + 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals