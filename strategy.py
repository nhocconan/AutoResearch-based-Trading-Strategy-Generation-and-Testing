#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian channel breakout with 1d ATR filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based volatility filter.
- Donchian channels: 20-period high/low from prior 4h candles.
- Breakout: Close > upper band (long) or Close < lower band (short) with volume > 1.5x 20-period volume MA.
- Volatility filter: Only trade when 1d ATR(14) is above its 50-period MA (avoid low-volatility chop).
- Works in bull via buying breakouts in expansion, in bear via selling breakdowns in expansion.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) and its 50-period MA for volatility filter
    # ATR = TR smoothed, where TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - prev_close)
    tr3 = abs(df_1d['low'] - prev_close)
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > atr_ma_50  # High volatility regime
    
    # Align 1d ATR filter to 4h
    atr_filter_aligned = align_htf_to_ltf(prices, df_1d, atr_filter)
    
    # Calculate 4h Donchian channels (20-period)
    # Use rolling window on prior candles only (no look-ahead)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian + ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(atr_filter_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Check for Donchian breakout with volume spike and volatility filter
            if volume_spike[i] and atr_filter_aligned[i]:
                # Long breakout: close > upper Donchian band
                if close[i] > donch_high[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakdown: close < lower Donchian band
                elif close[i] < donch_low[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price re-enters Donchian channel or opposite signal
            if close[i] < donch_low[i]:  # Exit when price falls below lower band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters Donchian channel or opposite signal
            if close[i] > donch_high[i]:  # Exit when price rises above upper band
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dATRFilter_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0