#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter, volume confirmation (1.5x MA20), and ATR volatility filter (ATR14 > 0.3 * ATR50).
# Enters long when price breaks above Donchian upper band with 1w bullish trend (close > EMA50), volume > 1.5x MA20, and sufficient volatility.
# Enters short when price breaks below Donchian lower band with 1w bearish trend (close < EMA50), volume > 1.5x MA20, and sufficient volatility.
# Exits when price reverts to the Donchian midpoint or ATR-based stoploss (2.0 * ATR14 from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~7-25/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Donchian(20) provides clear price channels, while 1w EMA50 filter ensures alignment with weekly momentum.
# Volume threshold (1.5x) and volatility filter (0.3x) reduce false breakouts, improving signal quality in both bull and bear markets.

name = "1d_Donchian20_Breakout_1wTrend_Volume_Volatility_v1"
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
    
    # Get 1w data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Calculate EMA(50) on 1w close
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels from previous 1d bar
    # Donchian(20) uses previous 20 days' high/low
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Donchian upper and lower bands (20-period)
    upper = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    midpoint = (upper + lower) / 2.0
    
    # Align Donchian levels to 1d timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower)
    midpoint_aligned = align_htf_to_ltf(prices, df_1d, midpoint)
    
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
        if np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or np.isnan(midpoint_aligned[i]) or \
           np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band with 1w bullish trend, volume spike, and sufficient volatility
            if close[i] > upper_aligned[i] and close[i] > ema50_1w_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Donchian lower band with 1w bearish trend, volume spike, and sufficient volatility
            elif close[i] < lower_aligned[i] and close[i] < ema50_1w_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Donchian midpoint (mean reversion) OR ATR stoploss hit
            if close[i] < midpoint_aligned[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Donchian midpoint (mean reversion) OR ATR stoploss hit
            if close[i] > midpoint_aligned[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals