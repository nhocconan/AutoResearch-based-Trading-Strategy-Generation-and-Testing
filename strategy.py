# Hypothesis: 6h timeframe with weekly pivot levels combined with ADX trend filter and volume confirmation.
# Weekly pivots provide strong institutional support/resistance levels that work in both bull and bear markets.
# ADX > 25 ensures we only trade in trending conditions, reducing whipsaw.
# Volume confirmation ensures breakouts have institutional participation.
# Target: 50-150 trades over 4 years (12-37/year) with position size 0.25.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # Pivot point = (H + L + C) / 3
    pp_w = (high_w + low_w + close_w) / 3
    # Resistance 1 = (2 * PP) - L
    r1_w = (2 * pp_w) - low_w
    # Support 1 = (2 * PP) - H
    s1_w = (2 * pp_w) - high_w
    # Resistance 2 = PP + (H - L)
    r2_w = pp_w + (high_w - low_w)
    # Support 2 = PP - (H - L)
    s2_w = pp_w - (high_w - low_w)
    
    # Align weekly pivot levels to 6h timeframe (need previous week's values)
    pp_w_aligned = align_htf_to_ltf(prices, df_1w, pp_w)
    r1_w_aligned = align_htf_to_ltf(prices, df_1w, r1_w)
    s1_w_aligned = align_htf_to_ltf(prices, df_1w, s1_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    
    # Calculate 14-period ADX for trend strength (using daily data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Wilder's smoothing
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        alpha = 1.0 / period
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    atr_1d = wilder_smooth(tr, 14)
    di_plus_1d = wilder_smooth(dm_plus, 14)
    di_minus_1d = wilder_smooth(dm_minus, 14)
    
    # Avoid division by zero
    di_sum = di_plus_1d + di_minus_1d
    dx = np.where(di_sum != 0, 100 * np.abs(di_plus_1d - di_minus_1d) / di_sum, 0)
    adx_1d = wilder_smooth(dx, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate daily average volume (20-period)
    volume_1d = df_1d['volume'].values
    vol_avg_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20)
    
    # Session filter: 8-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Session filter: only trade 8-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = prices['close'].iloc[i]
        adx_val = adx_1d_aligned[i]
        vol_val = prices['volume'].iloc[i]
        vol_avg_val = vol_avg_20_aligned[i]
        pp_val = pp_w_aligned[i]
        r1_val = r1_w_aligned[i]
        s1_val = s1_w_aligned[i]
        r2_val = r2_w_aligned[i]
        s2_val = s2_w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(adx_val) or np.isnan(vol_avg_val) or 
            np.isnan(pp_val) or np.isnan(r1_val) or np.isnan(s1_val) or
            np.isnan(r2_val) or np.isnan(s2_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: ADX > 25 (trending), price above weekly pivot and R1, volume above average
            if adx_val > 25 and close_val > pp_val and close_val > r1_val and vol_val > vol_avg_val:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending), price below weekly pivot and S1, volume above average
            elif adx_val > 25 and close_val < pp_val and close_val < s1_val and vol_val > vol_avg_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below weekly pivot or S1, or ADX < 20 (trend weakening)
            if close_val < pp_val or close_val < s1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above weekly pivot or R1, or ADX < 20 (trend weakening)
            if close_val > pp_val or close_val > r1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_WeeklyPivot_ADX_Volume_Session
# Uses weekly pivot points for institutional support/resistance levels
# Uses daily ADX for trend strength filter (ADX > 25)
# Requires volume confirmation above 20-day average
# Session filter: 8-20 UTC to avoid low-volume periods
# Exits when price crosses weekly pivot/support/resistance or trend weakens (ADX < 20)
# Designed for 6h timeframe with ~15-35 trades/year
name = "6h_WeeklyPivot_ADX_Volume_Session"
timeframe = "6h"
leverage = 1.0