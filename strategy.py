#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + volume spike + ATR filter.
- Long when price breaks above upper Donchian(20) AND volume > 2.0 * 20-period average AND ATR(14) < 0.03 * close
- Short when price breaks below lower Donchian(20) AND volume > 2.0 * 20-period average AND ATR(14) < 0.03 * close
- Exit when price crosses opposite Donchian band or ATR(14) > 0.05 * close (volatility expansion)
- Uses 4h primary timeframe for optimal trade frequency (target: 20-50 trades/year)
- Donchian channels provide clear structure, volume confirms conviction, ATR filter avoids choppy markets
- Works in both bull markets (breakouts continuation) and bear markets (breakdown continuation)
- Signal size: 0.30 discrete levels
- Target: 75-200 total trades over 4 years (19-50/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian(20) channels
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    upper_donchian = rolling_max(high, 20)
    lower_donchian = rolling_min(low, 20)
    
    # Calculate ATR(14) for volatility filter
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr1[0] = 0
        tr2[0] = 0
        tr3[0] = 0
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = np.zeros_like(tr)
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    # ATR filter: avoid high volatility environments
    atr_filter = atr_14 < (0.03 * close)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 20)  # Donchian(20), ATR(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper_donchian[i]) or np.isnan(lower_donchian[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND volume confirmation AND ATR filter
            if close[i] > upper_donchian[i] and volume_confirm[i] and atr_filter[i]:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below lower Donchian AND volume confirmation AND ATR filter
            elif close[i] < lower_donchian[i] and volume_confirm[i] and atr_filter[i]:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower Donchian OR ATR expansion (volatility > 5% of price)
            if close[i] < lower_donchian[i] or atr_14[i] > (0.05 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Short exit: price crosses above upper Donchian OR ATR expansion
            if close[i] > upper_donchian[i] or atr_14[i] > (0.05 * close[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRFilter_v1"
timeframe = "4h"
leverage = 1.0