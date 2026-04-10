#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Primary: 6h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13)
# - HTF: 1d ADX(14) > 25 for trending regime (avoid ranging markets)
# - Long: Bull Power > 0 and Bear Power < 0 (strong bullish momentum) + ADX > 25
# - Short: Bear Power < 0 and Bull Power < 0 (strong bearish momentum) + ADX > 25
# - Exit: Opposite Elder Ray signal (Bear Power > 0 for long exit, Bull Power > 0 for short exit)
# - Position sizing: 0.25 (discrete level, manages drawdown from 77% crashes)
# - Works in bull/bear: ADX regime filter ensures we only trade strong trends, Elder Ray captures momentum
# - Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_1d_elderray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for calculations
        return np.zeros(n)
    
    # Pre-compute 6h data
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 6h EMA13 for Elder Ray
    close_s_6h = pd.Series(close_6h)
    ema13_6h = close_s_6h.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 6h Elder Ray components
    bull_power_6h = high_6h - ema13_6h  # Bull Power = High - EMA13
    bear_power_6h = low_6h - ema13_6h   # Bear Power = Low - EMA13
    
    # Calculate 1d ADX(14)
    # True Range
    tr_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # +DM and -DM
    plus_dm_1d = np.full(len(close_1d), np.nan)
    minus_dm_1d = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        high_diff = high_1d[i] - high_1d[i-1]
        low_diff = low_1d[i-1] - low_1d[i]
        if high_diff > low_diff and high_diff > 0:
            plus_dm_1d[i] = high_diff
        elif low_diff > high_diff and low_diff > 0:
            minus_dm_1d[i] = low_diff
    
    # Smoothed TR, +DM, -DM (Wilder's smoothing)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[1:period])  # Skip index 0 as it's NaN
        # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] - (result[i-1]/period) + arr[i]
        return result
    
    atr_1d = wilder_smooth(tr_1d, 14)
    plus_di_1d = 100 * wilder_smooth(plus_dm_1d, 14) / atr_1d
    minus_di_1d = 100 * wilder_smooth(minus_dm_1d, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilder_smooth(dx_1d, 14)
    
    # Align HTF indicators to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(bull_power_6h[i]) or np.isnan(bear_power_6h[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # ADX regime filter: ADX > 25 = trending market
        trending_regime = adx_1d_aligned[i] > 25
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Bull Power > 0 and Bear Power < 0 (strong bullish) + trending regime
            if bull_power_6h[i] > 0 and bear_power_6h[i] < 0 and trending_regime:
                position = 1
                signals[i] = 0.25
            # Short entry: Bear Power < 0 and Bull Power < 0 (strong bearish) + trending regime
            elif bear_power_6h[i] < 0 and bull_power_6h[i] < 0 and trending_regime:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit: Opposite Elder Ray signal
            if position == 1:  # Long position
                if bear_power_6h[i] > 0:  # Bear power turned positive - exit long
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short position)
                if bull_power_6h[i] > 0:  # Bull power turned positive - exit short
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
    
    return signals