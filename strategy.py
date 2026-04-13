#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_v3
Hypothesis: Uses Camarilla pivot levels from daily timeframe for breakout entries on 4h chart.
In ranging markets, price often reverses from H3/L3 levels; in trending markets, breaks through H4/L4 indicate strong momentum.
Combines with volume confirmation and ADX trend filter to avoid false breakouts.
Works in both bull and bear markets by adapting to regime: mean reversion at H3/L3 in ranging markets (ADX<25), breakout at H4/L4 in trending markets (ADX>25).
Target: 20-50 trades/year on 4h (80-200 total over 4 years).
"""

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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.1 * (high - low)
    # L3 = close - 1.1 * (high - low)
    # H2 = close + 0.55 * (high - low)
    # L2 = close - 0.55 * (high - low)
    # H1 = close + 0.275 * (high - low)
    # L1 = close - 0.275 * (high - low)
    # Pivot = (high + low + close) / 3
    
    range_1d = high_1d - low_1d
    H4 = close_1d + 1.5 * range_1d
    L4 = close_1d - 1.5 * range_1d
    H3 = close_1d + 1.1 * range_1d
    L3 = close_1d - 1.1 * range_1d
    H2 = close_1d + 0.55 * range_1d
    L2 = close_1d - 0.55 * range_1d
    H1 = close_1d + 0.275 * range_1d
    L1 = close_1d - 0.275 * range_1d
    Pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Get 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate ADX for trend filtering on 4h
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] > minus_dm[i]:
                minus_dm[i] = 0
            elif minus_dm[i] > plus_dm[i]:
                plus_dm[i] = 0
            else:
                plus_dm[i] = 0
                minus_dm[i] = 0
                
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Smooth using Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        
        atr[period] = np.mean(tr[1:period+1])
        plus_dm_smoothed = np.zeros_like(plus_dm)
        minus_dm_smoothed = np.zeros_like(minus_dm)
        
        plus_dm_smoothed[period] = np.mean(plus_dm[1:period+1])
        minus_dm_smoothed[period] = np.mean(minus_dm[1:period+1])
        
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_smoothed[i] = (plus_dm_smoothed[i-1] * (period-1) + plus_dm[i]) / period
            minus_dm_smoothed[i] = (minus_dm_smoothed[i-1] * (period-1) + minus_dm[i]) / period
            
            if atr[i] != 0:
                plus_di[i] = 100 * plus_dm_smoothed[i] / atr[i]
                minus_di[i] = 100 * minus_dm_smoothed[i] / atr[i]
            else:
                plus_di[i] = 0
                minus_di[i] = 0
        
        dx = np.zeros_like(high)
        adx = np.zeros_like(high)
        
        for i in range(period*2, len(high)):
            if plus_di[i] + minus_di[i] != 0:
                dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100
            else:
                dx[i] = 0
        
        adx[2*period] = np.mean(dx[period:2*period])
        for i in range(2*period+1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
            
        return adx
    
    adx = calculate_adx(high_4h, low_4h, close_4h, 14)
    
    # Align all daily levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    adx_aligned = align_htf_to_ltf(prices, df_4h, adx)
    
    # Volume confirmation on 4h
    vol_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean()
    volume_spike = volume_4h > (vol_ma_20_4h * 1.5)
    volume_spike_aligned = align_htf_to_ltf(prices, df_4h, volume_spike.values)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or \
           np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or \
           np.isnan(Pivot_aligned[i]) or np.isnan(adx_aligned[i]) or \
           np.isnan(volume_spike_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine market regime: ADX > 25 = trending, ADX < 25 = ranging
        is_trending = adx_aligned[i] > 25
        
        # Entry logic based on regime
        if is_trending:
            # Trending market: breakout through H4/L4 with volume
            if close[i] > H4_aligned[i] and volume_spike_aligned[i]:
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif close[i] < L4_aligned[i] and volume_spike_aligned[i]:
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            # Hold existing positions
            elif position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # Ranging market: mean reversion from H3/L3 levels
            if close[i] < H3_aligned[i] and close[i] > L3_aligned[i]:
                # Inside H3-L3 range, look for mean reversion signals
                if close[i] <= L3_aligned[i] * 1.005 and position != 1:  # Near L3, go long
                    position = 1
                    signals[i] = position_size
                elif close[i] >= H3_aligned[i] * 0.995 and position != -1:  # Near H3, go short
                    position = -1
                    signals[i] = -position_size
                elif position == 1:
                    signals[i] = position_size
                elif position == -1:
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                # Outside H3-L3, wait for reversion back in
                if position == 1:
                    signals[i] = position_size
                elif position == -1:
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_v3"
timeframe = "4h"
leverage = 1.0