#!/usr/bin/env python3
"""
4h_pullback_trend_continuation_v1
Hypothesis: On 4h timeframe, enter long on pullbacks to the 20-period EMA during strong uptrends (price above 50-period EMA and ADX > 25), and enter short on pullbacks to the 20-period EMA during strong downtrends (price below 50-period EMA and ADX > 25). Use 1d ADX as trend strength filter to avoid weak trends. Exit when price crosses the 20-period EMA in the opposite direction. Designed for 20-40 trades/year to minimize fee decay while capturing trend continuation moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_pullback_trend_continuation_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period EMA for entry zone
    ema_20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Calculate 50-period EMA for trend filter
    ema_50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate ADX (14-period) for trend strength
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus14 / tr14
    di_minus = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ADX for trend strength filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    tr_1d[0] = tr1_1d[0]
    
    # Directional Movement for 1d
    dm_plus_1d = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                          np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus_1d = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                           np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus_1d[0] = 0
    dm_minus_1d[0] = 0
    
    # Smoothed values for 1d
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    dm_plus14_1d = pd.Series(dm_plus_1d).rolling(window=14, min_periods=14).sum().values
    dm_minus14_1d = pd.Series(dm_minus_1d).rolling(window=14, min_periods=14).sum().values
    
    # DI+ and DI- for 1d
    di_plus_1d = 100 * dm_plus14_1d / tr14_1d
    di_minus_1d = 100 * dm_minus14_1d / tr14_1d
    
    # DX and ADX for 1d
    dx_1d = 100 * np.abs(di_plus_1d - di_minus_1d) / (di_plus_1d + di_minus_1d)
    adx_1d = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(ema_20[i]) or np.isnan(ema_50[i]) or np.isnan(adx[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below 20 EMA (trend weakening)
            if close[i] < ema_20[i] and close[i-1] >= ema_20[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above 20 EMA (trend weakening)
            if close[i] > ema_20[i] and close[i-1] <= ema_20[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Strong trend filter: ADX > 25 on both 4h and 1d
            strong_trend = adx[i] > 25 and adx_1d_aligned[i] > 25
            
            if strong_trend:
                # Long: pullback to 20 EMA during uptrend (price above 50 EMA)
                if close[i] >= ema_20[i] and close[i-1] < ema_20[i-1] and close[i] > ema_50[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: pullback to 20 EMA during downtrend (price below 50 EMA)
                elif close[i] <= ema_20[i] and close[i-1] > ema_20[i-1] and close[i] < ema_50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals