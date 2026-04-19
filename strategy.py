#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_ADX_Filter
Hypothesis: 12h KAMA trend direction combined with ADX filter for strong trends.
KAMA adapts to market noise, reducing false signals in choppy markets.
ADX > 25 ensures we only trade in trending conditions, avoiding whipsaws.
Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
Works in bull/bear via adaptive trend following and volatility filtering.
"""

name = "12h_KAMA_Trend_With_ADX_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    fast_sc = 2 / (2 + 1)  # 2-period EMA smoothing constant
    slow_sc = 2 / (30 + 1) # 30-period EMA smoothing constant
    
    # Efficiency Ratio and KAMA calculation
    def calculate_kama(close, er_period=10, fast=2, slow=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly below
        
        # Proper ER calculation
        price_change = np.abs(close - np.roll(close, er_period))
        er_period_sum = np.zeros_like(close)
        for i in range(er_period, len(close)):
            er_period_sum[i] = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        
        # Avoid division by zero
        er = np.zeros_like(close)
        mask = er_period_sum > 0
        er[mask] = price_change[mask] / er_period_sum[mask]
        
        # Smoothing constant
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # ADX calculation (Wilder's smoothing)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Wilder's smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
                for i in range(period, len(data)):
                    if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                        result[i] = result[i-1] + alpha * (data[i] - result[i-1])
                    else:
                        result[i] = np.nan
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # DX calculation
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        # ADX
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate indicators on 12h data
    kama_12h = calculate_kama(df_12h['close'].values)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX filter: only trade when ADX > 25 (strong trend)
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price above KAMA with volume and strong trend
            if (close[i] > kama_aligned[i] and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and strong trend
            elif (close[i] < kama_aligned[i] and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or trend weakens
            if (close[i] < kama_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or trend weakens
            if (close[i] > kama_aligned[i]) or (adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals