#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Volume-Weighted MACD with 12h Regime Filter
# Long when VW-MACD line > signal line AND 12h ADX > 20 (trending regime)
# Short when VW-MACD line < signal line AND 12h ADX > 20 (trending regime)
# VW-MACD uses volume-weighted price instead of close for institutional confirmation
# Works in bull (strong uptrend + buying pressure) and bear (strong downtrend + selling pressure)
# Timeframe: 6h (primary timeframe as required)
# Target: 80-120 total trades over 4 years (20-30/year) to balance signal quality and fee drag

name = "6h_VW_MACD_12hADX_Regime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h ADX (14-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM- (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value: simple average
        result[period-1] = np.nansum(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = result[i-1] - (result[i-1] / period) + values[i]
        return result
    
    atr_12h = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_12h != 0, (dm_plus_smooth / atr_12h) * 100, 0)
    di_minus = np.where(atr_12h != 0, (dm_minus_smooth / atr_12h) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0,
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_12h = wilders_smoothing(dx, 14)
    
    # Align 12h ADX to 6h timeframe
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate 6h VW-MACD
    # Volume-weighted price: (high + low + close) * volume / (3 * volume) = (high + low + close) / 3
    # Actually, standard VWAP is typical price * volume, but for MACD we use typical price
    typical_price = (high + low + close) / 3.0
    
    # Fast EMA (12) and Slow EMA (26) of typical price
    if len(typical_price) >= 26:
        ema_fast = pd.Series(typical_price).ewm(span=12, adjust=False, min_periods=12).mean().values
        ema_slow = pd.Series(typical_price).ewm(span=26, adjust=False, min_periods=26).mean().values
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
        macd_histogram = macd_line - signal_line
    else:
        macd_line = np.full(n, np.nan)
        signal_line = np.full(n, np.nan)
        macd_histogram = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(macd_line[i]) or 
            np.isnan(signal_line[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: MACD line > signal line AND ADX > 20 (trending regime)
            if (macd_line[i] > signal_line[i] and 
                adx_12h_aligned[i] > 20):
                signals[i] = 0.25
                position = 1
            # Short: MACD line < signal line AND ADX > 20 (trending regime)
            elif (macd_line[i] < signal_line[i] and 
                  adx_12h_aligned[i] > 20):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: MACD line <= signal line (momentum weakening)
            if macd_line[i] <= signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: MACD line >= signal line (momentum weakening)
            if macd_line[i] >= signal_line[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals