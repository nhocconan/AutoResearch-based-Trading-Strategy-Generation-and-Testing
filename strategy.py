#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter, volume confirmation (1.5x MA20), and ATR(14) volatility filter.
# Enters long when price breaks above Donchian upper channel (20-period high) with 1d bullish trend (close > EMA50), volume > 1.5x MA20, and ATR(14) > 0.3 * ATR(50).
# Enters short when price breaks below Donchian lower channel (20-period low) with 1d bearish trend (close < EMA50), volume > 1.5x MA20, and ATR(14) > 0.3 * ATR(50).
# Exits when price reverts to the Donchian middle (10-period median of channel) or ATR-based stoploss (2 * ATR(14) from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~20-50/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Donchian channels provide clear structural breakouts, effective in both bull and bear markets when aligned with higher timeframe trend.
# The 1d trend filter ensures alignment with higher timeframe direction, while volatility filter avoids low volatility false breakouts.

name = "4h_Donchian20_Breakout_1dTrend_Volume_Volatility_v1"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian Channel (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0  # middle channel for exit
    
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
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper channel with 1d bullish trend, volume spike, and sufficient volatility
            if close[i] > donchian_high[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Donchian lower channel with 1d bearish trend, volume spike, and sufficient volatility
            elif close[i] < donchian_low[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Donchian middle (mean reversion) OR ATR stoploss hit
            if close[i] < donchian_mid[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Donchian middle (mean reversion) OR ATR stoploss hit
            if close[i] > donchian_mid[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals