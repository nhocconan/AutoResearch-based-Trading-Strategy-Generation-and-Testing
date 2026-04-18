#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly pivot point reversal with volume confirmation and ADX trend filter.
# Weekly pivot levels (R1, S1) provide key support/resistance from higher timeframe.
# Trade reversals at these levels with volume confirmation to avoid false breaks.
# ADX filter ensures we only trade in trending markets (ADX > 25) to avoid chop.
# Designed for low trade frequency (10-30/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (buy at S1 in uptrend) and bear markets (sell at R1 in downtrend).
name = "1d_WeeklyPivot_R1S1_Reversal_Volume_ADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points using previous week's data to avoid look-ahead
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Calculate pivot point and support/resistance levels
    pp = (high_w[:-1] + low_w[:-1] + close_w[:-1]) / 3  # previous week's data
    r1 = 2 * pp - low_w[:-1]
    s1 = 2 * pp - high_w[:-1]
    
    # Align weekly pivot levels to daily timeframe (already delayed by using previous week)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Calculate ADX (14-period) on daily data for trend strength
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smooth TR, DM+ and DM- using Wilder's smoothing (EMA with alpha=1/14)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Calculate DI+ and DI-
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # Calculate DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade in trending markets (ADX > 25)
        trend_filter = adx[i] > 25
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Long: price crosses above S1 AND volume confirmation AND trend filter
            long_signal = close[i] > s1_aligned[i] and close[i-1] <= s1_aligned[i-1]
            if vol_confirm and trend_filter and long_signal:
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1 AND volume confirmation AND trend filter
            elif vol_confirm and trend_filter and close[i] < r1_aligned[i] and close[i-1] >= r1_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below PP OR ADX drops below 20 (trend weakening)
            exit_signal = close[i] < pp_aligned[i] and close[i-1] >= pp_aligned[i-1]
            if exit_signal or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above PP OR ADX drops below 20 (trend weakening)
            exit_signal = close[i] > pp_aligned[i] and close[i-1] <= pp_aligned[i-1]
            if exit_signal or adx[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals