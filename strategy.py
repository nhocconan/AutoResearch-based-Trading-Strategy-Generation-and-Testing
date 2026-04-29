#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX Trend Filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
# Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (strong trend)
# Exit when Elder Ray signals weaken (Bull/Bear Power crosses zero)
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 12-37 trades/year on 6h timeframe.
# Elder Ray measures trend strength via price relative to EMA; ADX confirms trend regime.
# Works in bull via sustained Bull Power > 0, in bear via sustained Bear Power > 0.
# Novelty: Combines Elder Ray (price-EMA relationship) with ADX regime filter on 6h timeframe.

name = "6h_ElderRay_1dADX25_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Elder Ray on 6h timeframe
    ema13_6h = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13_6h  # Bull Power = High - EMA13
    bear_power = ema13_6h - low   # Bear Power = EMA13 - Low
    
    # Calculate ADX on 1d timeframe for trend filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr_1d = np.concatenate([[np.nan], tr])  # First value NaN
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus_1d = np.concatenate([[0], dm_plus])
    dm_minus_1d = np.concatenate([[0], dm_minus])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr_1d, period)
    dm_plus_smooth = wilders_smoothing(dm_plus_1d, period)
    dm_minus_smooth = wilders_smoothing(dm_minus_1d, period)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d != 0, dm_plus_smooth / atr_1d * 100, 0)
    di_minus = np.where(atr_1d != 0, dm_minus_smooth / atr_1d * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx_1d = wilders_smoothing(dx, period)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 50)  # Elder Ray EMA13 and ADX warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        curr_bull = bull_power[i]
        curr_bear = bear_power[i]
        curr_adx = adx_1d_aligned[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Elder Ray weakens (Bull Power <= 0 or Bear Power >= 0)
            if curr_bull <= 0 or curr_bear >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Elder Ray weakens (Bear Power <= 0 or Bull Power >= 0)
            if curr_bear <= 0 or curr_bull >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when Bull Power > 0 AND Bear Power < 0 AND 1d ADX > 25 (strong trend)
            if curr_bull > 0 and curr_bear < 0 and curr_adx > 25:
                signals[i] = 0.25
                position = 1
            # Short when Bear Power > 0 AND Bull Power < 0 AND 1d ADX > 25 (strong trend)
            elif curr_bear > 0 and curr_bull < 0 and curr_adx > 25:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals