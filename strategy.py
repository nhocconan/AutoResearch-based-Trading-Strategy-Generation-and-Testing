#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla Pivot + Volume Spike + Weekly ADX Trend Filter
# Camarilla pivot levels provide high-probability reversal zones (H3/L3).
# Volume spike confirms breakout/breakdown conviction.
# Weekly ADX ensures we only trade in strong trends, avoiding whipsaws.
# Designed for low trade frequency (10-25/year) to minimize fee drag in 1d timeframe.
# Works in bull markets (long at L3 breakout with volume + rising ADX) and bear markets 
# (short at H3 breakdown with volume + rising ADX).
name = "1d_Camarilla_H3L3_Volume_Spike_WeeklyADX"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    # Get weekly data for ADX filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla pivot levels using previous day's OHLC
    # Pivot point (PP) = (High + Low + Close) / 3
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = High - Low
    rng = df_1d['high'] - df_1d['low']
    
    # Resistance levels
    r1 = pp + (rng * 1.0833)  # H3 equivalent
    r2 = pp + (rng * 1.1666)  # H4
    r3 = pp + (rng * 1.2500)  # H5
    r4 = pp + (rng * 1.3333)  # H6
    
    # Support levels
    s1 = pp - (rng * 1.0833)  # L3 equivalent
    s2 = pp - (rng * 1.1666)  # L4
    s3 = pp - (rng * 1.2500)  # L5
    s4 = pp - (rng * 1.3333)  # L6
    
    # Focus on H3 (r1) and L3 (s1) as primary reversal zones
    h3 = r1.values
    l3 = s1.values
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    h3 = np.roll(h3, 1)
    l3 = np.roll(l3, 1)
    h3[0] = np.nan
    l3[0] = np.nan
    
    # Calculate weekly ADX for trend strength
    high_w = df_1w['high'].values
    low_w = df_1w['low'].values
    close_w = df_1w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_w[1:] - high_w[:-1]) > (low_w[:-1] - low_w[1:]), 
                       np.maximum(high_w[1:] - high_w[:-1], 0), 0)
    dm_minus = np.where((low_w[:-1] - low_w[1:]) > (high_w[1:] - high_w[:-1]), 
                        np.maximum(low_w[:-1] - low_w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing function
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nansum(data[:period]) / period
            # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] * (1 - 1/period) + data[i] * (1/period)
                else:
                    result[i] = np.nan
        return result
    
    atr_period = 14
    atr = wilders_smoothing(tr, atr_period)
    dm_plus_smooth = wilders_smoothing(dm_plus, atr_period)
    dm_minus_smooth = wilders_smoothing(dm_minus, atr_period)
    
    # DI values
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = wilders_smoothing(dx, atr_period)  # ADX is smoothed DX
    
    # Align weekly ADX to daily timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Calculate 20-period average volume for spike detection
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        if not vol_spike[i]:
            # No volume spike, maintain current position or flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Strong trend filter: ADX > 25
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: Close breaks above H3 with volume spike and strong trend
            if close[i] > h3[i] and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below L3 with volume spike and strong trend
            elif close[i] < l3[i] and strong_trend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: Close breaks below L3 (reversal signal)
            if close[i] < l3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close breaks above H3 (reversal signal)
            if close[i] > h3[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals