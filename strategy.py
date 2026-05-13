#!/usr/bin/env python3
# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (EMA50) and volume confirmation (1.3x MA20).
# Enters long when price breaks above Camarilla R1 level with 4h bullish trend (close > EMA50) and volume > 1.3x MA20.
# Enters short when price breaks below Camarilla S1 level with 4h bearish trend (close < EMA50) and volume > 1.3x MA20.
# Exits when price reverts to the Camarilla pivot point.
# Uses discrete position sizing (0.20) to limit fee churn and manage drawdown.
# Designed for low trade frequency (~15-37/year) by requiring confluence: price breakout + HTF trend + volume confirmation.
# Camarilla levels provide intraday support/resistance effective in ranging markets, while 4h trend filter ensures alignment with higher timeframe direction.
# Volume confirmation avoids low-quality breakouts, improving signal consistency across market regimes.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume_v1"
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
    
    # Get 4h data for trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Calculate EMA(50) on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Get 1d data for Camarilla levels (previous day)
    df_1d = get_htf_data(prices, '1d')
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    r1 = pivot + (range_hl * 1.1 / 12.0)  # Resistance 1
    s1 = pivot - (range_hl * 1.1 / 12.0)  # Support 1
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume filter: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.3)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # open_time is datetime64[ms], index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or np.isnan(pivot_aligned[i]) or \
           np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_ma20[i]):
            signals[i] = 0.0
            continue
        
        if not session_filter[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 with 4h bullish trend and volume spike
            if close[i] > r1_aligned[i] and close[i] > ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 with 4h bearish trend and volume spike
            elif close[i] < s1_aligned[i] and close[i] < ema50_4h_aligned[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price reverts to Camarilla pivot (mean reversion)
            if close[i] < pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price reverts to Camarilla pivot (mean reversion)
            if close[i] > pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals