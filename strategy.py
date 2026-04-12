#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_camarilla_breakout_v41
# Camarilla pivot levels from 1-day chart with volume confirmation and ADX trend filter.
# Uses ADX > 25 to confirm trending markets and avoid false signals in ranging conditions.
# Long when price breaks above H4 resistance with volume confirmation in uptrend.
# Short when price breaks below L4 support with volume confirmation in downtrend.
# Designed for fewer trades (target 20-30/year) to minimize fee drag and improve generalization.
name = "4h_1d_camarilla_breakout_v41"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Camarilla formulas
    range_prev = high_prev - low_prev
    camarilla_h4 = close_prev + range_prev * 1.1 / 2
    camarilla_l4 = close_prev - range_prev * 1.1 / 2
    
    # Align to 4h timeframe (already delayed by 1 day due to shift)
    h4_level = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_level = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    # ADX trend filter: only trade in trending markets (ADX > 25)
    # Calculate True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Calculate Directional Movement
    dm_plus = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), np.maximum(high[1:] - high[:-1], 0), 0)
    dm_minus = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), np.maximum(low[:-1] - low[1:], 0), 0)
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to RMA)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current_value
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # Avoid division by zero
    dm_plus_smooth = np.where(dm_plus_smooth == 0, 1e-10, dm_plus_smooth)
    dm_minus_smooth = np.where(dm_minus_smooth == 0, 1e-10, dm_minus_smooth)
    atr_safe = np.where(atr == 0, 1e-10, atr)
    
    di_plus = 100 * dm_plus_smooth / atr_safe
    di_minus = 100 * dm_minus_smooth / atr_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    trend_filter = adx_aligned > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # start after warmup
        # Skip if levels not ready
        if np.isnan(h4_level[i]) or np.isnan(l4_level[i]) or np.isnan(adx_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Check volume and trend filters
        if not (vol_confirm[i] and trend_filter[i]):
            # Hold current position if filters fail
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above H4 with volume in uptrend
        if close[i] > h4_level[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below L4 with volume in downtrend
        elif close[i] < l4_level[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite breakout
        elif close[i] < l4_level[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > h4_level[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals