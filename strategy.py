# Hypothesis: 12h Camarilla Pivot Reversal with Volume Spike and ADX Trend Filter
# Uses 12h Camarilla pivot levels for reversal signals, volume spike (>2.0x) for confirmation,
# and ADX(14) < 20 to avoid choppy markets. Long at S1 with volume spike in weak trend,
# short at R1 with volume spike in weak trend. Exit at opposite pivot level.
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability reversals.

#!/usr/bin/env python3
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    # Camarilla formulas: 
    # H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    # H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # H2 = close + 1.1*(high-low)/6, L2 = close - 1.1*(high-low)/6
    # H1 = close + 1.1*(high-low)/12, L1 = close - 1.1*(high-low)/12
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot levels for each 12h bar (based on previous bar's OHLC)
    r1_12h = np.full(len(df_12h), np.nan)
    s1_12h = np.full(len(df_12h), np.nan)
    r2_12h = np.full(len(df_12h), np.nan)
    s2_12h = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        range_val = prev_high - prev_low
        
        if range_val > 0:
            r1_12h[i] = prev_close + 1.1 * range_val / 12
            s1_12h[i] = prev_close - 1.1 * range_val / 12
            r2_12h[i] = prev_close + 1.1 * range_val / 6
            s2_12h[i] = prev_close - 1.1 * range_val / 6
    
    # Align pivot levels to 15m timeframe (assuming 15m base, but works with any)
    r1_12h_aligned = align_htf_to_ltf(prices, df_12h, r1_12h)
    s1_12h_aligned = align_htf_to_ltf(prices, df_12h, s1_12h)
    r2_12h_aligned = align_htf_to_ltf(prices, df_12h, r2_12h)
    s2_12h_aligned = align_htf_to_ltf(prices, df_12h, s2_12h)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX(14) on daily data
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([np.array([np.nan]), tr])  # Align with index
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([np.array([0]), dm_plus])
    dm_minus = np.concatenate([np.array([0]), dm_minus])
    
    # Smooth TR, DM+, DM- with Wilder's smoothing (alpha = 1/14)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr > 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr > 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to lower timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume spike detection: > 2.0x 24-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 24
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup period
    start_idx = max(1, vol_period)  # Need at least 1 for pivot, 24 for volume
    
    for i in range(start_idx, n):
        if (np.isnan(r1_12h_aligned[i]) or
            np.isnan(s1_12h_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Weak trend filter: ADX < 20 (avoid strong trends where reversals fail)
        weak_trend = adx_aligned[i] < 20
        
        # Volume confirmation: spike > 2.0x average
        volume_confirmation = vol_ratio > 2.0
        
        if position == 0:
            # Long reversal at S1: price touches S1 with volume spike in weak trend
            if weak_trend and volume_confirmation and price <= s1_12h_aligned[i] * 1.001:  # Allow small slippage
                signals[i] = size
                position = 1
            # Short reversal at R1: price touches R1 with volume spike in weak trend
            elif weak_trend and volume_confirmation and price >= r1_12h_aligned[i] * 0.999:  # Allow small slippage
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price reaches R1 (take profit) or R2 (stop)
            if price >= r1_12h_aligned[i] * 0.999:  # Take profit at R1
                signals[i] = 0.0
                position = 0
            elif price >= r2_12h_aligned[i] * 0.999:  # Stop at R2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price reaches S1 (take profit) or S2 (stop)
            if price <= s1_12h_aligned[i] * 1.001:  # Take profit at S1
                signals[i] = 0.0
                position = 0
            elif price <= s2_12h_aligned[i] * 1.001:  # Stop at S2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_Reversal_ADX_Volume"
timeframe = "12h"
leverage = 1.0