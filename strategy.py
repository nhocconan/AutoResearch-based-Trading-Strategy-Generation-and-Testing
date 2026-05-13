#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation (1.8x MA30).
# Enters long when price breaks above Camarilla R3 level with 1d bullish trend (close > EMA50) and volume > 1.8x MA30.
# Enters short when price breaks below Camarilla S3 level with 1d bearish trend (close < EMA50) and volume > 1.8x MA30.
# Exits when price reverts to Camarilla pivot point (PP) or ATR-based stoploss hit (2.0 * ATR20 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels provide intraday support/resistance structure, while 1d EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.8x) reduces false breakouts, improving signal quality in both bull and bear markets.

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v1"
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
    
    # Get 1d data for HTF indicators
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    # PP = (H + L + C) / 3
    pp_1d = (high_1d + low_1d + close_1d) / 3.0
    # R3 = PP + (H - L) * 1.1 / 2
    r3_1d = pp_1d + (high_1d - low_1d) * 1.1 / 2.0
    # S3 = PP - (H - L) * 1.1 / 2
    s3_1d = pp_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align Camarilla levels to 12h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # Volume filter: current volume > 1.8x 30-period average
    volume_series = pd.Series(volume)
    vol_ma30 = volume_series.rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma30 * 1.8)
    
    # ATR(20) for stoploss
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr20 = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma30[i]) or \
           np.isnan(atr20[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot point (PP) OR ATR stoploss hit
            if close[i] < pp_aligned[i] or close[i] < entry_price[i-1] - 2.0 * atr20[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot point (PP) OR ATR stoploss hit
            if close[i] > pp_aligned[i] or close[i] > entry_price[i-1] + 2.0 * atr20[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals