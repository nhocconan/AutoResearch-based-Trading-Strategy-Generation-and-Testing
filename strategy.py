#!/usr/bin/env python3
# Hypothesis: 1d Camarilla R3/S3 breakout with 1w EMA34 trend filter, volume confirmation (1.5x MA20), and ATR-based volatility filter (ATR14 > 0.3 * ATR50).
# Enters long when price breaks above Camarilla R3 level with 1w bullish trend (close > EMA34), volume > 1.5x MA20, and sufficient volatility.
# Enters short when price breaks below Camarilla S3 level with 1w bearish trend (close < EMA34), volume > 1.5x MA20, and sufficient volatility.
# Exits when price reverts to the Camarilla midpoint (HLC/3) or ATR-based stoploss (2.0 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~15-25/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Camarilla levels provide clear support/resistance structure, while 1w EMA34 filter ensures alignment with higher timeframe momentum.
# Volume threshold (1.5x) and volatility filter (0.3x) reduce false breakouts, improving signal quality in both bull and bear markets.

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume_Volatility_v1"
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
    
    # Calculate Camarilla levels (R3, S3, midpoint) from 1d data
    # Camarilla: based on previous day's OHLC
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # Midpoint = (high + low + close) / 3
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    # First bar: use current values as fallback
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) / 2.0
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) / 2.0
    camarilla_mid = (prev_high + prev_low + prev_close) / 3.0
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr14 > (0.3 * atr50)  # avoid low volatility breakouts
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(camarilla_r3[i]) or np.isnan(camarilla_s3[i]) or np.isnan(camarilla_mid[i]) or \
           np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R3 with 1w bullish trend, volume spike, and sufficient volatility
            if close[i] > camarilla_r3[i] and close[i] > ema34_1w_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S3 with 1w bearish trend, volume spike, and sufficient volatility
            elif close[i] < camarilla_s3[i] and close[i] < ema34_1w_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla midpoint (mean reversion) OR ATR stoploss hit
            if close[i] < camarilla_mid[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla midpoint (mean reversion) OR ATR stoploss hit
            if close[i] > camarilla_mid[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals