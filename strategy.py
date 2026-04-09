#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) + 1d ADX trend filter + volume confirmation
# Elder Ray measures bull/bear strength relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Strong Bull Power + rising ADX (>25) = strong uptrend momentum for long entries
# Strong Bear Power + rising ADX (>25) = strong downtrend momentum for short entries
# Volume confirmation (1.5x 20-period avg) filters weak signals
# Works in bull/bear: ADX ensures we only trade strong trends, avoiding ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25-0.30

name = "6h_1d_elder_ray_adx_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(values, period):
        result = np.full_like(values, np.nan)
        if len(values) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(values[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    period = 14
    atr_1d = wilders_smoothing(tr, period)
    dm_plus_smooth = wilders_smoothing(dm_plus, period)
    dm_minus_smooth = wilders_smoothing(dm_minus, period)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1d, np.nan)
    di_minus = np.full_like(atr_1d, np.nan)
    mask = ~np.isnan(atr_1d) & (atr_1d != 0)
    di_plus[mask] = 100 * dm_plus_smooth[mask] / atr_1d[mask]
    di_minus[mask] = 100 * dm_minus_smooth[mask] / atr_1d[mask]
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan)
    mask = ~np.isnan(di_plus) & ~np.isnan(di_minus) & ((di_plus + di_minus) != 0)
    dx[mask] = 100 * np.abs(di_plus[mask] - di_minus[mask]) / (di_plus[mask] + di_minus[mask])
    
    adx_1d = wilders_smoothing(dx, period)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Smooth Elder Ray signals (optional, reduces noise)
    bull_power_smooth = pd.Series(bull_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=5, adjust=False, min_periods=5).mean().values
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or
            np.isnan(bear_power_smooth[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        # ADX trend filter: only trade when trend is strong (ADX > 25)
        strong_trend = adx_1d_aligned[i] > 25
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes negative (momentum loss) OR ADX weakens
            if bear_power_smooth[i] > 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative (momentum loss) OR ADX weakens
            if bull_power_smooth[i] < 0 or not strong_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, strong trend, and Elder Ray signals
            if volume_confirmed and strong_trend:
                # Long entry: Strong Bull Power (positive) AND strong trend
                if bull_power_smooth[i] > 0:
                    position = 1
                    signals[i] = 0.25
                # Short entry: Strong Bear Power (negative) AND strong trend
                elif bear_power_smooth[i] < 0:
                    position = -1
                    signals[i] = -0.25
    
    return signals