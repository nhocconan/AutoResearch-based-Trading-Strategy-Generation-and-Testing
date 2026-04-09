#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d ADX regime filter
# Williams Alligator (JAW=13, TEETH=8, LIPS=5) identifies trend direction and strength
# 1d ADX > 25 filters for trending markets only, reducing whipsaws in ranging conditions
# Entry: Alligator aligned (Lips > Teeth > Jaw for long, reverse for short) + ADX > 25
# Exit: Alligator misaligned or ADX < 20 (hysteresis to prevent chatter)
# Uses discrete position sizing 0.25 to target ~12-37 trades/year and minimize fee drag
# Works in bull/bear markets: ADX regime filter ensures we only trade strong trends

name = "6h_1d_williams_alligator_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14) with proper Wilder's smoothing
    def wilders_smoothing(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        alpha = 1.0 / period
        result = np.full(len(values), np.nan)
        result[period-1] = np.nanmean(values[:period])
        for i in range(period, len(values)):
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
        return result
    
    # True Range calculation
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+
    atr_1d = wilders_smoothing(tr, 14)
    dm_plus_smooth = wilders_smoothing(dm_plus, 14)
    dm_minus_smooth = wilders_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.where(atr_1d > 0, 100 * dm_plus_smooth / atr_1d, 0)
    di_minus = np.where(atr_1d > 0, 100 * dm_minus_smooth / atr_1d, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1d = wilders_smoothing(dx, 14)
    
    # Williams Alligator on 6h timeframe
    # JAW (Blue): 13-period SMMA, shifted 8 bars forward
    # TEETH (Red): 8-period SMMA, shifted 5 bars forward  
    # LIPS (Green): 5-period SMMA, shifted 3 bars forward
    def smma(values, period):
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        result[period-1] = np.mean(values[:period])
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    # Calculate SMMA for median price (typical price)
    typical_price = (high + low + close) / 3
    jaw = smma(typical_price, 13)
    teeth = smma(typical_price, 8)
    lips = smma(typical_price, 5)
    
    # Apply Alligator shifts (jaw: +8, teeth: +5, lips: +3)
    jaw_shifted = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips_shifted = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Align 1d ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 for trending market (enter), ADX < 20 to exit (hysteresis)
        strong_trend = adx_1d_aligned[i] > 25
        weak_trend = adx_1d_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit if Alligator misaligned or trend weakens
            if not (lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]) or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit if Alligator misaligned or trend weakens
            if not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]) or weak_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long if Alligator aligned for up and strong trend
            if lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i] and strong_trend:
                position = 1
                signals[i] = 0.25
            # Enter short if Alligator aligned for down and strong trend
            elif lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and strong_trend:
                position = -1
                signals[i] = -0.25
    
    return signals