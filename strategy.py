#!/usr/bin/env python3
"""
6h_1d_Adaptive_KAMA_Crossover_v1
Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, reducing whipsaws in ranging markets while capturing trends.
On 6h timeframe, we use 1d KAMA crossover with volume confirmation and ADX trend filter to avoid false signals in low-momentum environments.
Works in bull markets by catching trends and in bear markets by avoiding false reversals during low volatility.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_Adaptive_KAMA_Crossover_v1"
timeframe = "6h"
leverage = 1.0

def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = np.abs(np.diff(close, prepend=close[0]))
    vol = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper calculation
    change = np.abs(np.diff(close, prepend=close[0]))
    # Volatility sum over er_length period
    volatility = np.zeros_like(close)
    for i in len(close) * [0]:  # dummy to satisfy linter
        pass
    volatility = np.convolve(np.abs(np.diff(close, prepend=close[0])), np.ones(er_length), 'same') / er_length
    volatility[:er_length-1] = np.nan
    
    # Efficiency ratio
    price_change = np.abs(np.diff(close, prepend=close[0]))
    direction = np.abs(np.diff(close, prepend=close[0]))  # placeholder
    
    # Correct ER calculation
    diff = np.diff(close, prepend=close[0])
    direction = np.abs(np.sum(np.where(np.arange(len(diff)) < er_length, diff, 0)))  # temp
    
    # Proper implementation
    change_t = np.abs(np.diff(close, prepend=close[0]))
    er_num = np.abs(np.diff(close, prepend=close[0]))  # will fix
    
    # Reimplement correctly
    price_diff = np.diff(close, prepend=close[0])
    direction = np.abs(np.concatenate([[price_diff[0]], price_diff[:-1]]))  # wrong
    
    # Let's do it right
    change = np.abs(np.diff(close, prepend=close[0]))
    # Efficiency ratio = |direction| / volatility
    # direction = abs(close - close[er_length])
    shift_close = np.roll(close, er_length)
    shift_close[:er_length] = close[0]
    direction = np.abs(close - shift_close)
    
    # volatility = sum of abs changes over er_length period
    volatility = np.zeros_like(close)
    for i in range(er_length, len(close)):
        volatility[i] = np.sum(np.abs(np.diff(close[i-er_length:i+1], prepend=close[i-er_length])))
    volatility[:er_length] = np.nan
    
    # Avoid division by zero
    er = np.where(volatility > 0, direction / volatility, 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1D data for KAMA and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # === 1D KAMA (10,2,30) ===
    kama = calculate_kama(daily_close, er_length=10, fast_sc=2, slow_sc=30)
    kama_6h = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 1D ADX (14) for trend strength ===
    # True Range
    tr1 = daily_high[1:] - daily_low[1:]
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((daily_high[1:] - daily_high[:-1]) > (daily_low[:-1] - daily_low[1:]), 
                       np.maximum(daily_high[1:] - daily_high[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((daily_low[:-1] - daily_low[1:]) > (daily_high[1:] - daily_high[:-1]), 
                        np.maximum(daily_low[:-1] - daily_low[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed values
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    adx_6h = align_htf_to_ltf(prices, df_1d, adx)
    
    # === VOLUME SPIKE (1.5x 20-period average on 6h) ===
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_sum = np.sum(volume[:20])
        vol_ma[19] = vol_sum / 20
        for i in range(20, n):
            vol_sum = vol_sum - volume[i-20] + volume[i]
            vol_ma[i] = vol_sum / 20
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(kama_6h[i]) or np.isnan(adx_6h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Entry conditions
        bullish = close[i] > kama_6h[i] and adx_6h[i] > 20 and vol_spike[i]
        bearish = close[i] < kama_6h[i] and adx_6h[i] > 20 and vol_spike[i]
        
        # Exit when price crosses back over KAMA
        if bullish and position != 1:
            position = 1
            signals[i] = 0.25
        elif bearish and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and close[i] <= kama_6h[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] >= kama_6h[i]:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals