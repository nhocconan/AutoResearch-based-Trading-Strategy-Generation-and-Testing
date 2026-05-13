#!/usr/bin/env python3
# Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation (1.8x MA20).
# Enters long when price breaks above Camarilla R3 level with 1d bullish trend (close > EMA34) and volume > 1.8x MA20.
# Enters short when price breaks below Camarilla S3 level with 1d bearish trend (close < EMA34) and volume > 1.8x MA20.
# Exits when price reverts to Camarilla H4/L4 midpoint or ATR-based stoploss hit (2.0 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence: price breakout at key pivot + HTF trend + volume spike.
# Camarilla pivots provide high-probability reversal/breakout levels, while 1d EMA34 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.8x) reduces false breakouts, improving signal quality in both bull and bear markets.
# ATR stoploss (2.0) allows for volatility adaptation while preventing excessive whipsaw.

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume_v2"
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
    
    # Get 1d data for trend filter (EMA34) and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(34) on 1d close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla pivot levels from 1d OHLC
    camarilla_h5 = (high_1d / low_1d) * close_1d
    camarilla_h4 = ((high_1d - low_1d) * 1.1 / 2) + close_1d
    camarilla_h3 = ((high_1d - low_1d) * 1.1 / 4) + close_1d
    camarilla_l3 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_l2 = close_1d - ((high_1d - low_1d) * 1.1 / 2)
    camarilla_l1 = close_1d - ((high_1d - low_1d) * 1.1 / 4)
    camarilla_h6 = camarilla_h5 + (camarilla_h5 - camarilla_h4)
    camarilla_s3 = camarilla_l3
    camarilla_r3 = camarilla_h3
    camarilla_h4_val = camarilla_h4
    camarilla_l4_val = camarilla_l2
    camarilla_mid = (camarilla_h4_val + camarilla_l4_val) / 2.0
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
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
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or \
           np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla H4/L4 midpoint OR ATR stoploss hit
            if close[i] < camarilla_mid_aligned[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla H4/L4 midpoint OR ATR stoploss hit
            if close[i] > camarilla_mid_aligned[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals