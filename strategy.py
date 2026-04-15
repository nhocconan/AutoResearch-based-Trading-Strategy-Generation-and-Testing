#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + volume confirmation + ADX trend filter
# Uses Donchian channel breakouts from 4h price action, confirmed by volume spikes
# and filtered by ADX to ensure trending markets. Works in both bull and bear
# markets by capturing strong directional moves while avoiding choppy conditions.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for ADX calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14-period)
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
        
        atr = np.zeros_like(high)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period).mean().values
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate volume average (20-period on 1d)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Calculate Donchian channels (20-period on 4h)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = calculate_donchian(high_4h, low_4h, 20)
    
    # Align all indicators to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Base position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i])):
            continue
        
        # Long entry: Price breaks above Donchian upper + volume spike + ADX > 25 (trending)
        if (close[i] > donchian_upper_aligned[i] and 
            volume[i] > 1.5 * vol_avg_aligned[i] and 
            adx_aligned[i] > 25 and 
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Price breaks below Donchian lower + volume spike + ADX > 25 (trending)
        elif (close[i] < donchian_lower_aligned[i] and 
              volume[i] > 1.5 * vol_avg_aligned[i] and 
              adx_aligned[i] > 25 and 
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: reverse signal or ADX falls below 20 (losing trend strength)
        elif position == 1 and adx_aligned[i] < 20:
            position = 0
            signals[i] = 0.0
        elif position == -1 and adx_aligned[i] < 20:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0