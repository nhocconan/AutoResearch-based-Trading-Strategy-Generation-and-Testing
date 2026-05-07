#!/usr/bin/env python3
"""
6h_ADX_Trend_With_Volume_Confirmation
Hypothesis: Uses ADX(14) > 25 to identify strong trends on 6h timeframe, with volume confirmation (volume > 20-period average) to filter false breakouts. Direction determined by +DI vs -DI crossover. Includes 1-day trend filter (price above/below EMA34) to align with higher timeframe bias. Designed for 12-30 trades/year to avoid fee drag while capturing trending moves in both bull and bear markets.
"""

name = "6h_ADX_Trend_With_Volume_Confirmation"
timeframe = "6h"
leverage = 1.0

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
    
    # Calculate ADX components
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    
    for i in range(1, n):
        high_diff = high[i] - high[i-1]
        low_diff = low[i-1] - low[i]
        
        plus_dm[i] = high_diff if high_diff > low_diff and high_diff > 0 else 0
        minus_dm[i] = low_diff if low_diff > high_diff and low_diff > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # Smooth with Wilder's smoothing (equivalent to EMA with alpha=1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
        return result
    
    period = 14
    atr = WilderSmoothing(tr, period)
    plus_di = 100 * WilderSmoothing(plus_dm, period) / atr
    minus_di = 100 * WilderSmoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmoothing(dx, period)
    
    # 1-day trend filter: EMA34 of daily close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: current volume > 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after ADX calculation is valid
    start_idx = 3 * period  # Need enough data for Wilder smoothing
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(adx[i]) or np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25, +DI > -DI, price above 1-day EMA, volume confirmation
            if (adx[i] > 25 and plus_di[i] > minus_di[i] and 
                close[i] > ema_1d_aligned[i] and volume[i] > vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25, -DI > +DI, price below 1-day EMA, volume confirmation
            elif (adx[i] > 25 and minus_di[i] > plus_di[i] and 
                  close[i] < ema_1d_aligned[i] and volume[i] > vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: ADX drops below 20 OR -DI crosses above +DI OR price below 1-day EMA
            if (adx[i] < 20 or minus_di[i] > plus_di[i] or 
                close[i] < ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: ADX drops below 20 OR +DI crosses above -DI OR price above 1-day EMA
            if (adx[i] < 20 or plus_di[i] > minus_di[i] or 
                close[i] > ema_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals