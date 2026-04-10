#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d volume confirmation + 1w ADX regime filter
# - Williams Alligator: Jaw (13-period SMMA smoothed 8), Teeth (8-period SMMA smoothed 5), Lips (5-period SMMA smoothed 3)
# - Long when Lips > Teeth > Jaw (bullish alignment) AND 1d volume > 1.5x 20-period average AND 1w ADX > 25
# - Short when Lips < Teeth < Jaw (bearish alignment) AND 1d volume > 1.5x 20-period average AND 1w ADX > 25
# - Exit when Alligator lines intertwine (Lips crosses Teeth or Jaw) or ADX < 20 (trend weakening)
# - Uses discrete position sizing 0.25 to limit fee churn
# - Williams Alligator identifies trend initiation and continuation with built-in smoothing
# - Volume confirmation ensures breakouts have participation
# - Weekly ADX filter ensures we trade only when higher timeframe is strongly trending
# - Target: 12-30 trades/year on 12h timeframe (50-120 total over 4 years)

name = "12h_1d_1w_alligator_volume_adx_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h OHLCV
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute Williams Alligator on 12h timeframe
    def smma(arr, period):
        """Smoothed Moving Average (Wilder's smoothing)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev * (period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Alligator components: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(smma(close, 13), 8)
    teeth = smma(smma(close, 8), 5)
    lips = smma(smma(close, 5), 3)
    
    # Pre-compute 1d volume average (20-period)
    volume_1d = df_1d['volume'].values
    vol_ma_1d = np.full_like(volume_1d, np.nan, dtype=float)
    for i in range(19, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-19:i+1])
    
    # Pre-compute 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Wilder's smoothing
    def wheilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        result[period-1] = np.nansum(arr[:period])
        for i in range(period, len(arr)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] - (result[i-1] / period) + arr[i]
        return result
    
    atr_1w = wheilder_smoothing(tr, 14)
    dm_plus_smooth = wheilder_smoothing(dm_plus, 14)
    dm_minus_smooth = wheilder_smoothing(dm_minus, 14)
    
    # DI+ and DI-
    di_plus = np.full_like(atr_1w, np.nan, dtype=float)
    di_minus = np.full_like(atr_1w, np.nan, dtype=float)
    for i in range(14, len(atr_1w)):
        if not np.isnan(atr_1w[i]) and atr_1w[i] != 0:
            di_plus[i] = (dm_plus_smooth[i] / atr_1w[i]) * 100
            di_minus[i] = (dm_minus_smooth[i] / atr_1w[i]) * 100
    
    # DX and ADX
    dx = np.full_like(di_plus, np.nan, dtype=float)
    for i in range(14, len(di_plus)):
        if not np.isnan(di_plus[i]) and not np.isnan(di_minus[i]):
            di_sum = di_plus[i] + di_minus[i]
            if di_sum != 0:
                dx[i] = np.abs(di_plus[i] - di_minus[i]) / di_sum * 100
    
    adx_1w = wheilder_smoothing(dx, 14)
    
    # Align HTF indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw) if '1d' in str(type(df_1d)) else jaw
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth) if '1d' in str(type(df_1d)) else teeth
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips) if '1d' in str(type(df_1d)) else lips
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Actually, we need to compute Alligator on 12h data directly, not 1d
    # Recompute Alligator on actual 12h prices
    jaw_12h = smma(smma(close, 13), 8)
    teeth_12h = smma(smma(close, 8), 5)
    lips_12h = smma(smma(close, 5), 3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(lips_12h[i]) or np.isnan(teeth_12h[i]) or np.isnan(jaw_12h[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(adx_1w_aligned[i])):
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:  # Flat - look for new entries
            # Volume spike condition (1.5x average)
            vol_ma_12h = np.full_like(volume, np.nan, dtype=float)
            for j in range(19, i+1):
                vol_ma_12h[j] = np.mean(volume[j-19:j+1])
            vol_spike = not np.isnan(vol_ma_12h[i]) and volume[i] > 1.5 * vol_ma_12h[i]
            
            # Bullish alignment: Lips > Teeth > Jaw
            bullish = lips_12h[i] > teeth_12h[i] and teeth_12h[i] > jaw_12h[i]
            # Bearish alignment: Lips < Teeth < Jaw
            bearish = lips_12h[i] < teeth_12h[i] and teeth_12h[i] < jaw_12h[i]
            
            # Long conditions: bullish alignment AND volume spike AND 1w trending (ADX > 25)
            if bullish and vol_spike and adx_1w_aligned[i] > 25:
                position = 1
                signals[i] = 0.25
            # Short conditions: bearish alignment AND volume spike AND 1w trending (ADX > 25)
            elif bearish and vol_spike and adx_1w_aligned[i] > 25:
                position = -1
                signals[i] = -0.25
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit conditions: Alligator lines intertwine (Lips crosses Teeth or Jaw) or ADX < 20
            lips_cross_teeth = (position == 1 and lips_12h[i] < teeth_12h[i]) or \
                              (position == -1 and lips_12h[i] > teeth_12h[i])
            lips_cross_jaw = (position == 1 and lips_12h[i] < jaw_12h[i]) or \
                            (position == -1 and lips_12h[i] > jaw_12h[i])
            trend_weakening = adx_1w_aligned[i] < 20
            
            if lips_cross_teeth or lips_cross_jaw or trend_weakening:
                position = 0
                signals[i] = 0.0
            else:
                if position == 1:
                    signals[i] = 0.25
                else:
                    signals[i] = -0.25
    
    return signals

def smma(arr, period):
    """Smoothed Moving Average (Wilder's smoothing)"""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value: simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev * (period-1) + current) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def wheilder_smoothing(arr, period):
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    result[period-1] = np.nansum(arr[:period])
    for i in range(period, len(arr)):
        if not np.isnan(result[i-1]):
            result[i] = result[i-1] - (result[i-1] / period) + arr[i]
    return result