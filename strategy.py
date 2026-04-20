# 6h_Cam_Piv_R1S1_Breakout_Volume
# Hypothesis: Fade at Camarilla R1/S1 when price reverts to mean in ranging markets (identified by low ADX < 20), and breakout continuation when price breaks R1/S1 in trending markets (ADX > 25).
# Uses 12h Camarilla levels for structure, 60-period ADX for regime, and volume spike for confirmation.
# Designed for 6h timeframe with ~15-25 trades/year, avoiding overtrading while capturing both mean reversion and trend continuation.
# Works in bull/bear by adapting to market regime via ADX.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for Camarilla levels and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    
    # Calculate 12h Camarilla pivot levels (R1, S1)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Typical price for pivot calculation
    typical_price = (high_12h + low_12h + close_12h) / 3.0
    # Calculate pivot point
    pivot = (high_12h + low_12h + close_12h) / 3.0
    # Calculate range
    range_12h = high_12h - low_12h
    # Camarilla levels
    r1 = pivot + (range_12h * 1.1 / 12)
    s1 = pivot - (range_12h * 1.1 / 12)
    
    # Align Camarilla levels to 6h
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # Calculate 12h ADX(14) for regime detection
    # True Range
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smooth TR, DM+, DM- using Wilder's smoothing (equivalent to EMA with alpha=1/14)
    def wilder_smooth(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(arr[1:period])
        # Subsequent values: smoothed = prev * (1 - 1/period) + current * (1/period)
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]) and not np.isnan(arr[i]):
                result[i] = result[i-1] * (1 - 1/period) + arr[i] * (1/period)
            else:
                result[i] = np.nan
        return result
    
    atr = wilder_smooth(tr, 14)
    dm_plus_smooth = wilder_smooth(dm_plus, 14)
    dm_minus_smooth = wilder_smooth(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr != 0, dm_plus_smooth / atr * 100, 0)
    di_minus = np.where(atr != 0, dm_minus_smooth / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    # ADX is smoothed DX
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Calculate 6h volume average for confirmation
    volume_6h = prices['volume'].values
    vol_avg_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Get values
        close_val = prices['close'].iloc[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        adx_val = adx_aligned[i]
        vol_val = volume_6h[i]
        vol_avg_val = vol_avg_20[i]
        
        # Skip if any value is NaN
        if (np.isnan(r1_val) or np.isnan(s1_val) or 
            np.isnan(adx_val) or np.isnan(vol_avg_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry:
            # Ranging market (ADX < 20): fade at R1/S1 (mean reversion)
            # Trending market (ADX > 25): breakout continuation at R1/S1
            if adx_val < 20:
                # Ranging: sell at R1, buy at S1
                if close_val <= r1_val and close_val > s1_val:
                    # Near R1 - consider short if rejecting
                    if close_val < r1_val * 0.999:  # Slight rejection from R1
                        signals[i] = -0.25
                        position = -1
                elif close_val >= s1_val and close_val < r1_val:
                    # Near S1 - consider long if rejecting
                    if close_val > s1_val * 1.001:  # Slight rejection from S1
                        signals[i] = 0.25
                        position = 1
            else:
                # Trending: breakout continuation
                if close_val > r1_val and vol_val > vol_avg_val * 1.5:
                    signals[i] = 0.25
                    position = 1
                elif close_val < s1_val and vol_val > vol_avg_val * 1.5:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: 
            # - Price reaches opposite level (S1) 
            # - Or ADX drops indicating trend weakness
            if close_val <= s1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit:
            # - Price reaches opposite level (R1)
            # - Or ADX drops indicating trend weakness
            if close_val >= r1_val or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# 6h_Cam_Piv_R1S1_Breakout_Volume
# Uses 12h Camarilla R1/S1 levels for structure
# Regime detection via 12h ADX(14): 
#   ADX < 20 = ranging (fade at R1/S1)
#   ADX > 25 = trending (breakout continuation at R1/S1)
# Volume confirmation: requires 1.5x average volume for breakouts
# Designed for 6h timeframe with ~15-25 trades/year
name = "6h_Cam_Piv_R1S1_Breakout_Volume"
timeframe = "6h"
leverage = 1.0