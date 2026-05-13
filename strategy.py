#!/usr/bin/env python3
# Hypothesis: 1h Donchian(20) breakout with 4h EMA50 trend filter, volume confirmation (1.5x MA20), and ATR(14) > 0.3*ATR(50) volatility filter.
# Enters long when price breaks above Donchian upper channel with 4h bullish trend (close > EMA50), volume > 1.5x MA20, and ATR(14) > 0.3 * ATR(50).
# Enters short when price breaks below Donchian lower channel with 4h bearish trend (close < EMA50), volume > 1.5x MA20, and ATR(14) > 0.3 * ATR(50).
# Exits when price crosses the Donchian midpoint or ATR-based stoploss (2 * ATR(14) from entry).
# Uses discrete position sizing (0.20) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~15-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Uses 4h for signal direction and 1h only for entry timing to avoid overtrading.
# Session filter (08-20 UTC) to reduce noise trades.

name = "1h_Donchian_Breakout_4hTrend_Volume_Volatility_v1"
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
    open_time = prices['open_time'].values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Donchian Channel (20) on 1h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
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
        # Skip if any required data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]) or \
           np.isnan(in_session[i]):
            signals[i] = 0.0
            continue
        
        # Only trade during session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper with 4h bullish trend, volume spike, and sufficient volatility
            if close[i] > highest_high[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.20
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Donchian lower with 4h bearish trend, volume spike, and sufficient volatility
            elif close[i] < lowest_low[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.20
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian midpoint OR ATR stoploss hit
            if close[i] < donchian_mid[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.20
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian midpoint OR ATR stoploss hit
            if close[i] > donchian_mid[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.20
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals