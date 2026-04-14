#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversals with 1d volume spike and 1d ADX trend filter
# Long when price touches Camarilla S3 level AND 1d volume > 2x average AND 1d ADX > 25 (trending)
# Short when price touches Camarilla R3 level AND 1d volume > 2x average AND 1d ADX > 25
# Exit when price reaches Camarilla C level (pivot) or opposite S3/R3 touch occurs
# Camarilla levels provide precise intraday support/resistance; volume confirms institutional interest; ADX filters for trending markets
# Designed to capture reversals in both bull and bear markets by fading extremes with trend confirmation
# Target: 50-150 total trades over 4 years (12-38/year) to minimize fee drag while capturing meaningful moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for volume, ADX, and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla formula: 
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.6*(high-low)
    # H1 = close + 0.375*(high-low)
    # L1 = close - 0.375*(high-low)
    # L2 = close - 0.6*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    # Pivot (C) = (high + low + close) / 3
    
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Calculate pivot and ranges
    pivot = (phigh + plow + pclose) / 3
    range_hl = phigh - plow
    
    # Camarilla levels
    S1 = pclose - 0.375 * range_hl
    S2 = pclose - 0.6 * range_hl
    S3 = pclose - 1.1 * range_hl
    S4 = pclose - 1.5 * range_hl
    R1 = pclose + 0.375 * range_hl
    R2 = pclose + 0.6 * range_hl
    R3 = pclose + 1.1 * range_hl
    R4 = pclose + 1.5 * range_hl
    C = pivot  # Central pivot
    
    # Calculate 1d ADX (14-period) for trend filter
    # ADX calculation: +DM, -DM, TR, then DX, then smoothed ADX
    high_diff = np.diff(phigh, prepend=phigh[0])
    low_diff = np.diff(plow, prepend=plow[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = phigh - plow
    tr2 = np.abs(phigh - np.roll(pclose, 1))
    tr3 = np.abs(plow - np.roll(pclose, 1))
    tr2[0] = np.abs(phigh[0] - pclose[0])
    tr3[0] = np.abs(plow[0] - pclose[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.zeros_like(data)
        result[period-1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    period = 14
    if len(tr) >= period:
        atr = wilders_smooth(tr, period)
        plus_di = 100 * wilders_smooth(plus_dm, period) / atr
        minus_di = 100 * wilders_smooth(minus_dm, period) / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        # Handle division by zero
        dx = np.where((plus_di + minus_di) != 0, dx, 0)
        adx = wilders_smooth(dx, period)
    else:
        adx = np.zeros_like(phigh)
    
    # Calculate 1d volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all 1d indicators to 4h timeframe
    S3_1d = align_htf_to_ltf(prices, df_1d, S3)
    R3_1d = align_htf_to_ltf(prices, df_1d, R3)
    C_1d = align_htf_to_ltf(prices, df_1d, C)
    adx_1d = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(S3_1d[i]) or np.isnan(R3_1d[i]) or np.isnan(C_1d[i]) or 
            np.isnan(adx_1d[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol = volume[i]
        vol_threshold = vol_avg_1d_aligned[i] * 2.0  # 2x average volume
        
        if position == 0:
            # Long setup: price touches S3 AND volume spike AND ADX > 25 (trending)
            if (low_val <= S3_1d[i] and vol > vol_threshold and adx_1d[i] > 25):
                position = 1
                signals[i] = position_size
            # Short setup: price touches R3 AND volume spike AND ADX > 25 (trending)
            elif (high_val >= R3_1d[i] and vol > vol_threshold and adx_1d[i] > 25):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches C level (pivot) OR touches R3 (extreme)
            if (close_val >= C_1d[i] or high_val >= R3_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches C level (pivot) OR touches S3 (extreme)
            if (close_val <= C_1d[i] or low_val <= S3_1d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_1dVolume_ADX"
timeframe = "4h"
leverage = 1.0