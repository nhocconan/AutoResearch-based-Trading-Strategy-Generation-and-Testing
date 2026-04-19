#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h volume confirmation and ADX trend filter.
# Works in both bull and bear markets by trading breakouts in the direction of the higher timeframe trend.
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost.
name = "4h_Donchian20_12hVolume_ADXFilter"
timeframe = "4h"
leverage = 1.0

def calculate_adx(high, low, close, period=14):
    plus_dm = np.zeros_like(high)
    minus_dm = np.zeros_like(high)
    tr = np.zeros_like(high)
    
    for i in range(1, len(high)):
        plus_dm[i] = max(high[i] - high[i-1], 0) if (high[i] - high[i-1]) > (low[i-1] - low[i]) else 0
        minus_dm[i] = max(low[i-1] - low[i], 0) if (low[i-1] - low[i]) > (high[i] - high[i-1]) else 0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=period, min_periods=period).mean().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=period, min_periods=period).mean().values
    tr_smooth = pd.Series(tr).rolling(window=period, min_periods=period).mean().values
    
    plus_di = 100 * plus_dm_smooth / (tr_smooth + 1e-10)
    minus_di = 100 * minus_dm_smooth / (tr_smooth + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=period, min_periods=period).mean().values
    return adx

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume confirmation (called ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume moving average
    volume_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    # Calculate ADX on 4h for trend strength
    adx = calculate_adx(high, low, close, 14)
    
    # Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(high_max[i]) or np.isnan(low_min[i]) or np.isnan(adx[i]) or np.isnan(volume_12h_ma_aligned[i]):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: current 12h volume > 1.5 * 20-period average
        volume_confirm = volume_12h[i // 12] > (volume_12h_ma_aligned[i] * 1.5) if i >= 12 else False
        
        # ADX filter: trend strength > 25
        trend_filter = adx[i] > 25
        
        if position == 0:
            # Long when price breaks above upper Donchian band with volume and trend confirmation
            if close[i] > high_max[i] and volume_confirm and trend_filter:
                signals[i] = 0.25
                position = 1
            # Short when price breaks below lower Donchian band with volume and trend confirmation
            elif close[i] < low_min[i] and volume_confirm and trend_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price breaks below lower Donchian band
            if close[i] < low_min[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price breaks above upper Donchian band
            if close[i] > high_max[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals