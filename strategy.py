#!/usr/bin/env python3
"""
12h_Pivot_R1S1_Breakout_Volume_ADX_Filter_v3
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and ADX trend filter
- Camarilla levels from 1D provide significant support/resistance
- Volume > 1.5x 20-period average confirms institutional participation
- ADX > 25 filters for trending markets to avoid false breakouts in chop
- Uses 1W EMA200 filter to align with long-term trend (bullish above, bearish below)
- Tight entry conditions target 50-150 total trades over 4 years
- Works in bull/bear via trend filters and volatility-adjusted stops
"""

name = "12h_Pivot_R1S1_Breakout_Volume_ADX_Filter_v3"
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
    
    # EMA200 on 1W for long-term trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    ema200_1w = pd.Series(df_1w['close'].values).ewm(span=200, adjust=False).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # ADX(14) for trend strength filter on 12H
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate ADX using Wilder's smoothing
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
        
        dx = np.full_like(close, np.nan)
        mask = (atr > 0) & ~np.isnan(atr) & ~np.isnan(dm_plus_smooth) & ~np.isnan(dm_minus_smooth)
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = WilderSmooth(dx, period)
        return adx
    
    adx_12h = calculate_adx(df_12h['high'].values, df_12h['low'].values, df_12h['close'].values, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Previous day's Camarilla levels (using 1D data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ph = df_1d['high'].shift(1).values
    pl = df_1d['low'].shift(1).values
    pc = df_1d['close'].shift(1).values
    
    rang = ph - pl
    r1 = pc + (rang * 1.1 / 12)
    s1 = pc - (rang * 1.1 / 12)
    r4 = pc + (rang * 1.1 / 2)
    s4 = pc - (rang * 1.1 / 2)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filters: ADX > 25 (strong trend) and price vs 1W EMA200
        strong_trend = adx_12h_aligned[i] > 25
        bullish_long_term = close[i] > ema200_1w_aligned[i]
        bearish_long_term = close[i] < ema200_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume, strong trend, and bullish long-term bias
            if (close[i] > r1_aligned[i] and 
                volume_confirm[i] and 
                strong_trend and 
                bullish_long_term):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume, strong trend, and bearish long-term bias
            elif (close[i] < s1_aligned[i] and 
                  volume_confirm[i] and 
                  strong_trend and 
                  bearish_long_term):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below S1 or long-term trend turns bearish
            if (close[i] < s1_aligned[i]) or (not bullish_long_term):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above R1 or long-term trend turns bullish
            if (close[i] > r1_aligned[i]) or (not bearish_long_term):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals