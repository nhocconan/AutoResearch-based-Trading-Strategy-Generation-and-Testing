#!/usr/bin/env python3
"""
4h_ADX_KAMA_Trend_Filter_V1
Hypothesis: 4h KAMA trend direction with ADX filter to avoid choppy markets.
KAMA adapts to market noise - effective in both trending and ranging conditions.
ADX > 25 ensures we only trade in strong trends, reducing false signals.
Designed for 4h timeframe targeting 20-50 trades/year (80-200 total over 4 years).
Works in bull/bear via adaptive trend following and volatility filtering.
"""

name = "4h_ADX_KAMA_Trend_Filter_V1"
timeframe = "4h"
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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close, prepend=close[0]))
        for i in range(1, len(volatility)):
            volatility[i] = volatility[i-1] + np.abs(close[i] - close[i-1])
        
        er = np.where(volatility > 0, change / volatility, 0)
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.full_like(close, np.nan)
        kama[0] = close[0]
        for i in range(1, len(close)):
            if not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = close[i]
        return kama
    
    # Calculate ADX for trend strength
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
        
        # Wilder smoothing
        def WilderSmooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] + (1/period) * (data[i] - result[i-1])
            return result
        
        atr = WilderSmooth(tr, period)
        dm_plus_smooth = WilderSmooth(dm_plus, period)
        dm_minus_smooth = WilderSmooth(dm_minus, period)
        
        # DX calculation
        dx = np.full_like(close, np.nan)
        dm_sum = dm_plus_smooth + dm_minus_smooth
        mask = (dm_sum > 0) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / dm_sum[mask]
        
        # ADX
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get 4h data for KAMA and ADX
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate indicators on 4h data
    kama = calculate_kama(df_4h['close'].values, er_length=10, fast_sc=2, slow_sc=30)
    adx = calculate_adx(df_4h['high'].values, df_4h['low'].values, df_4h['close'].values, 14)
    
    # Align to lower timeframe (4h->4h is identity but keeps consistency)
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
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
        
        # ADX filter: only trade when ADX > 25 (trending market)
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