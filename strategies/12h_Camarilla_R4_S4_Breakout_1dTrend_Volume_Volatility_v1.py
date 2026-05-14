#!/usr/bin/env python3
# Hypothesis: 12h Camarilla R4/S4 breakout with 1d trend filter (close > EMA50), volume confirmation (1.8x MA20), and ATR volatility filter.
# Enters long when price breaks above Camarilla R4 level with 1d bullish trend (close > EMA50), volume > 1.8x MA20, and ATR(14) > 0.4 * ATR(50).
# Enters short when price breaks below Camarilla S4 level with 1d bearish trend (close < EMA50), volume > 1.8x MA20, and ATR(14) > 0.4 * ATR(50).
# Exits when price reverts to the Camarilla pivot point or ATR-based stoploss (2.5 * ATR(14) from entry).
# Uses discrete position sizing (0.28) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~12-37/year) by requiring strict confluence: price breakout + HTF trend + volume spike + volatility filter.
# Camarilla levels provide strong support/resistance, while HTF trend filter ensures alignment with higher timeframe direction.
# Higher volume threshold (1.8x) and volatility filter (0.4x) reduce false breakouts, improving signal quality in both bull and bear markets.

name = "12h_Camarilla_R4_S4_Breakout_1dTrend_Volume_Volatility_v1"
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
    
    # Get 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate EMA(50) on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla levels are calculated using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous day high
    prev_low = df_1d['low'].shift(1).values    # previous day low
    prev_close = df_1d['close'].shift(1).values # previous day close
    
    # Calculate Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r4 = pivot + (range_hl * 1.1 / 2.0)  # Resistance 4
    s4 = pivot - (range_hl * 1.1 / 2.0)  # Support 4
    r3 = pivot + (range_hl * 1.1 / 4.0)  # Resistance 3
    s3 = pivot - (range_hl * 1.1 / 4.0)  # Support 3
    
    # Align Camarilla levels to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume filter: current volume > 1.8x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.8)
    
    # ATR(14) and ATR(50) for volatility filter
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]  # first bar
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    volatility_filter = atr14 > (0.4 * atr50)  # avoid low volatility breakouts
    
    # Track entry price for ATR-based stoploss
    entry_price = np.full(n, np.nan)
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(pivot_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i]) or \
           np.isnan(atr14[i]) or np.isnan(atr50[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R4 with 1d bullish trend, volume spike, and sufficient volatility
            if close[i] > r4_aligned[i] and close[i] > ema50_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = 0.28
                position = 1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            # SHORT: Price breaks below Camarilla S4 with 1d bearish trend, volume spike, and sufficient volatility
            elif close[i] < s4_aligned[i] and close[i] < ema50_1d_aligned[i] and volume_spike[i] and volatility_filter[i]:
                signals[i] = -0.28
                position = -1
                entry_price[i] = close[i]  # record entry price at close of signal bar
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot (mean reversion) OR ATR stoploss hit
            if close[i] < pivot_aligned[i] or close[i] < entry_price[i-1] - 2.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = 0.28
                entry_price[i] = entry_price[i-1]  # carry forward entry price
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot (mean reversion) OR ATR stoploss hit
            if close[i] > pivot_aligned[i] or close[i] > entry_price[i-1] + 2.5 * atr14[i]:
                signals[i] = 0.0
                position = 0
                entry_price[i] = np.nan
            else:
                signals[i] = -0.28
                entry_price[i] = entry_price[i-1]  # carry forward entry price
    
    return signals