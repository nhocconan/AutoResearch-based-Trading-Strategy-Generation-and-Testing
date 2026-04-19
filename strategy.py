#!/usr/bin/env python3
# 12h_KAMA_Trend_Trader
# Hypothesis: Kaufman Adaptive Moving Average (KAMA) adapts to market noise, providing reliable trend signals.
# In trending markets (ADX > 25), KAMA captures strong moves with fewer whipsaws.
# In ranging markets (ADX < 20), KAMA flattens, reducing false signals.
# Combined with volume confirmation to ensure institutional participation.
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in bull/bear via adaptive trend filtering and volatility-adjusted signals.

name = "12h_KAMA_Trend_Trader"
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
    
    # KAMA (Kaufman Adaptive Moving Average) - faster adaptation to trend changes
    def calculate_kama(close, er_period=10, fast_sc=2, slow_sc=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, prepend=close[0]))
        abs_change = np.abs(np.diff(close, prepend=close[0]))
        # Alternative: sum of absolute changes over er_period
        er_num = np.abs(np.subtract(close[er_period:], close[:-er_period]))
        er_den = np.sum(np.abs(np.diff(close, prepend=close[0])[:len(close)-er_period+1]), axis=0) if len(close) > er_period else 1
        # Vectorized approach
        er = np.zeros_like(close)
        for i in range(er_period, len(close)):
            if i >= er_period:
                num = np.abs(close[i] - close[i-er_period])
                den = np.sum(np.abs(np.diff(close[i-er_period+1:i+1])))
                er[i] = num / den if den != 0 else 0
        er[:er_period] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        
        # KAMA calculation
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    # ADX for trend strength filter
    def calculate_adx(high, low, close, period=14):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
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
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    # Get 12h data for KAMA and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA and ADX on 12h data
    kama_12h = calculate_kama(df_12h['close'].values, er_period=8, fast_sc=2, slow_sc=30)
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: volume > 1.3 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(adx_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: ADX > 25 for strong trend, price relative to KAMA for direction
        strong_trend = adx_12h_aligned[i] > 25
        above_kama = close[i] > kama_12h_aligned[i]
        below_kama = close[i] < kama_12h_aligned[i]
        
        if position == 0:
            # Long: price above KAMA with volume and strong trend
            if (above_kama and 
                volume_confirm[i] and 
                strong_trend):
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA with volume and strong trend
            elif (below_kama and 
                  volume_confirm[i] and 
                  strong_trend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price crosses below KAMA or trend weakens (ADX < 20)
            if (below_kama) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price crosses above KAMA or trend weakens (ADX < 20)
            if (above_kama) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals