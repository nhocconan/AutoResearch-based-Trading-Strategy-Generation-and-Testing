#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with 1d ADX regime filter
# Bull Power = High - EMA13, Bear Power = EMA13 - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong trend)
# Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong trend)
# Exit when power values cross zero (momentum shift) or ADX < 20 (weak trend)
# Uses 1d EMA13 for power calculation and 1d ADX(14) for regime filter
# Works in bull markets by capturing strong uptrends and in bear markets by capturing strong downtrends
# ADX filter prevents whipsaws in ranging markets, focusing on high-probability trending moves
# Target: 12-25 trades/year on 6h timeframe (48-100 total over 4 years) to minimize fee drag

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for EMA13 and ADX calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power_1d = high_1d - ema_13_1d
    bear_power_1d = ema_13_1d - low_1d
    
    # Calculate 1d ADX(14) for regime filter
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # align with index 0
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d > 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilder_smooth(dx, 14)
    
    # Align all indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 14*2)  # EMA13 and ADX warmup (need ~28 bars for ADX)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_13_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_bull = bull_power_1d_aligned[i]
        curr_bear = bear_power_1d_aligned[i]
        curr_adx = adx_1d_aligned[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Bull Power <= 0 (momentum weakening) OR ADX < 20 (weak trend)
            if curr_bull <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power <= 0 (momentum weakening) OR ADX < 20 (weak trend)
            if curr_bear <= 0 or curr_adx < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND ADX > 25 (strong uptrend)
            if curr_bull > 0 and curr_bear < 0 and curr_adx > 25:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 AND Bull Power < 0 AND ADX > 25 (strong downtrend)
            elif curr_bear > 0 and curr_bull < 0 and curr_adx > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals