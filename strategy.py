#!/usr/bin/env python3
# 12h_KAMA_Direction_Trend_Confirmation
# Hypothesis: 12h Kaufman Adaptive Moving Average (KAMA) direction with trend confirmation (ADX > 25)
# KAMA adapts to market noise - stays flat in sideways markets, follows trend in trending markets
# Combined with ADX filter to only trade in strong trends, reducing false signals in chop
# Designed for 12h timeframe targeting 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear via trend-following nature and volatility adaptation

name = "12h_KAMA_Direction_Trend_Confirmation"
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
    
    # Calculate KAMA on 12h data
    def calculate_kama(close, er_length=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) == 1 else np.convolve(np.abs(np.diff(close)), np.ones(er_length), 'same')
        # Simplified ER calculation
        er = np.zeros_like(close)
        for i in range(er_length, len(close)):
            if np.sum(np.abs(np.diff(close[i-er_length:i+1]))) > 0:
                er[i] = np.abs(close[i] - close[i-er_length]) / np.sum(np.abs(np.diff(close[i-er_length:i+1])))
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # ADX calculation for trend strength
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
            alpha = 1.0 / period
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
        dx_sum = dm_plus_smooth + dm_minus_smooth
        mask = (dx_sum > 0) & ~np.isnan(dx_sum)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / dx_sum[mask]
        
        # ADX
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate indicators on 12h data
    kama_12h = calculate_kama(df_12h['close'].values, 10, 2, 30)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    # Align to lower timeframe
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation on 12h
    volume_ma = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_confirm = df_12h['volume'].values > (volume_ma * 1.5)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_12h, volume_confirm)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price above KAMA with volume and strong trend
            if (close[i] > kama_aligned[i] and 
                volume_confirm_aligned[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and strong trend
            elif (close[i] < kama_aligned[i] and 
                  volume_confirm_aligned[i] and 
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