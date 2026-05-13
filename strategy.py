#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator breakout with 1d trend filter (EMA50), volume confirmation (1.5x MA20), and ATR(14) volatility filter.
# Enters long when price breaks above Alligator's Jaw (SMMA13) with 1d bullish trend (close > EMA50), volume > 1.5x MA20, and ATR(14) > 0.3 * ATR(50).
# Enters short when price breaks below Alligator's Jaw (SMMA13) with 1d bearish trend (close < EMA50), volume > 1.5x MA20, and ATR(14) > 0.3 * ATR(50).
# Exits when price crosses the Alligator's Teeth (SMMA8) or ATR-based stoploss (2 * ATR(14) from entry).
# Uses discrete position sizing (0.25) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Williams Alligator uses smoothed moving averages (SMMA) that reduce whipsaw in ranging markets, effective in both bull and bear regimes.
# The 1d trend filter ensures alignment with higher timeframe direction, while volatility filter avoids low volatility false breakouts.

name = "12h_WilliamsAlligator_Breakout_1dTrend_Volume_Volatility_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(values) < period:
        return np.full(len(values), np.nan)
    result = np.full(len(values), np.nan)
    # First value is simple SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

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
    
    # Williams Alligator on 12h data: Jaw (SMMA13), Teeth (SMMA8), Lips (SMMA5)
    # Using smoothed moving averages (Wilder's smoothing)
    jaw = smma(close, 13)   # Alligator's Jaw (slowest)
    teeth = smma(close, 8)  # Alligator's Teeth (middle)
    lips = smma(close, 5)   # Alligator's Lips (fastest)
    
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
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Alligator's Jaw with 1d bullish trend, volume spike, and sufficient volatility
            if close[i] > jaw[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.25
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Alligator's Jaw with 1d bearish trend, volume spike, and sufficient volatility
            elif close[i] < jaw[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.25
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Alligator's Teeth (mean reversion) OR ATR stoploss hit
            if close[i] < teeth[i] or close[i] < entry_price[i-1] - 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price crosses above Alligator's Teeth (mean reversion) OR ATR stoploss hit
            if close[i] > teeth[i] or close[i] > entry_price[i-1] + 2.0 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.25
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals