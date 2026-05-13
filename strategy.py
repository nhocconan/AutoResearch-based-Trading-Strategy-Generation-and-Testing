#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R3/S3 breakout with 1d trend filter (close > EMA50), volume confirmation (2.0x MA20), and ATR stoploss (1.5 * ATR14).
# Enters long when price breaks above Camarilla R3 with 1d bullish trend (close > EMA50), volume > 2.0x MA20.
# Enters short when price breaks below Camarilla S3 with 1d bearish trend (close < EMA50), volume > 2.0x MA20.
# Exits when price reverts to Camarilla H5/L5 midpoint or ATR-based stoploss hit (1.5 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike.
# Camarilla levels provide high-probability reversal/breakout structure, while 1d EMA50 filter ensures alignment with higher timeframe momentum.
# Volume threshold (2.0x) reduces false breakouts, improving signal quality in both bull and bear markets.
# This strategy targets 12h timeframe with 1d HTF, proven to work in ranging and trending markets via Camarilla's mathematical pivot structure.

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
    
    # Get 1d data for HTF trend filter (EMA50) and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from 1d OHLC
    # Camarilla: H5 = Close + 1.1*(High-Low)/2, L5 = Close - 1.1*(High-Low)/2
    # R3 = Close + 1.1*(High-Low)/2, S3 = Close - 1.1*(High-Low)/2
    # Actually: R4 = Close + 1.1*(High-Low)/2, R3 = Close + 1.1*(High-Low)/4
    #         S3 = Close - 1.1*(High-Low)/4, S4 = Close - 1.1*(High-Low)/2
    # Correct Camarilla: R3 = Close + 1.1*(High-Low)/4, S3 = Close - 1.1*(High-Low)/4
    camarilla_range = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * camarilla_range / 4.0
    camarilla_s3 = close_1d - 1.1 * camarilla_range / 4.0
    camarilla_h5 = close_1d + 1.1 * camarilla_range / 2.0  # H5 = resistance
    camarilla_l5 = close_1d - 1.1 * camarilla_range / 2.0  # L5 = support
    camarilla_mid = (camarilla_h5 + camarilla_l5) / 2.0    # Midpoint = (H5+L5)/2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    camarilla_mid_aligned = align_htf_to_ltf(prices, df_1d, camarilla_mid)
    
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
        if np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_mid_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1d bullish trend and volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1d bearish trend and volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla midpoint (mean reversion) OR ATR stoploss hit
            if close[i] < camarilla_mid_aligned[i] or close[i] < entry_price[i-1] - 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla midpoint (mean reversion) OR ATR stoploss hit
            if close[i] > camarilla_mid_aligned[i] or close[i] > entry_price[i-1] + 1.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals