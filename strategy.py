#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with weekly pivot point (R1/S1) breakout + volume confirmation + weekly ADX trend filter
# Uses weekly pivot levels for structural support/resistance, volume to confirm breakout strength,
# and ADX to avoid ranging markets. Designed for fewer trades (target: 15-30/year) to minimize fee drag.
# Works in both bull/bear: breakouts capture trends, ADX filter avoids whipsaws in ranges.

name = "12h_1w_Pivot_R1S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 14:  # Need at least 14 weeks for ADX calculation
        return np.zeros(n)
    
    # === Weekly: Pivot Points (using prior week) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot points for current week using previous week's data
    pp = (np.roll(high_1w, 1) + np.roll(low_1w, 1) + np.roll(close_1w, 1)) / 3.0
    r1 = 2 * pp - np.roll(low_1w, 1)
    s1 = 2 * pp - np.roll(high_1w, 1)
    
    # Align to 12h timeframe (wait for weekly bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Weekly: ADX for trend strength ===
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(np.roll(high, 1) - close)
        tr3 = np.abs(np.roll(low, 1) - close)
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        
        # Smooth TR, DM+, DM- using Wilder's smoothing (alpha = 1/period)
        def wilder_smooth(arr, period):
            result = np.full_like(arr, np.nan)
            if len(arr) >= period:
                # First value is simple average
                result[period-1] = np.nansum(arr[:period]) / period
                # Subsequent values: Wilder smoothing
                for i in range(period, len(arr)):
                    result[i] = (result[i-1] * (period-1) + arr[i]) / period
            return result
        
        atr = wilder_smooth(tr, period)
        dm_plus_smooth = wilder_smooth(dm_plus, period)
        dm_minus_smooth = wilder_smooth(dm_minus, period)
        
        # Directional Indicators
        di_plus = 100 * dm_plus_smooth / np.where(atr > 0, atr, np.nan)
        di_minus = 100 * dm_minus_smooth / np.where(atr > 0, atr, np.nan)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) > 0, (di_plus + di_minus), np.nan)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # === 12h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Get values
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if np.isnan(r1_val) or np.isnan(s1_val) or np.isnan(adx_val) or np.isnan(vol_ratio_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R1 with volume and trend strength
            if (close_val > r1_val and          # Price breaks above weekly R1
                adx_val > 25 and                # Strong trend (ADX > 25)
                vol_ratio_val > 1.5):           # Volume confirmation
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume and trend strength
            elif (close_val < s1_val and        # Price breaks below weekly S1
                  adx_val > 25 and              # Strong trend (ADX > 25)
                  vol_ratio_val > 1.5):         # Volume confirmation
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns below pivot point or trend weakens
            if (close_val < pp_aligned[i] or    # Price returns below weekly pivot
                adx_val < 20):                  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns above pivot point or trend weakens
            if (close_val > pp_aligned[i] or    # Price returns above weekly pivot
                adx_val < 20):                  # Trend weakening
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals