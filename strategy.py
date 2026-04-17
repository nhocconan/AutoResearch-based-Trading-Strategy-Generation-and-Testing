#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d volume spike + ADX regime filter.
Long when price breaks above Donchian(20) high with 1d volume > 1.5x 20-period average and ADX > 25.
Short when price breaks below Donchian(20) low with 1d volume > 1.5x 20-period average and ADX > 25.
Exit when price touches opposite Donchian(20) level or ADX < 20 (range regime).
Uses 4h for price action and breakouts, 1d for volume and trend regime filter.
Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Calculate 4h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = calculate_donchian(high, low, 20)
    
    # Get 1d data for volume and ADX regime filters
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for Donchian and 1d indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(adx_14_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: 1d volume > 1.5x 20-period average
        volume_spike = volume_1d[i // 16] > 1.5 * vol_ma_20_aligned[i] if i // 16 < len(volume_1d) else False
        
        # Regime condition: ADX > 25 (trending market)
        is_trending = adx_14_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_up = close[i] > upper_20[i-1]  # Using previous close to avoid look-ahead
        breakout_down = close[i] < lower_20[i-1]
        
        # Exit conditions: touch opposite Donchian level or ADX < 20 (range)
        touch_upper = close[i] >= upper_20[i]
        touch_lower = close[i] <= lower_20[i]
        is_ranging = adx_14_aligned[i] < 20
        
        if position == 0:
            # Long: Donchian breakout up + volume spike + trending regime
            if breakout_up and volume_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down + volume spike + trending regime
            elif breakout_down and volume_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: touch lower Donchian or range regime
            if touch_lower or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: touch upper Donchian or range regime
            if touch_upper or is_ranging:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ADXRegime"
timeframe = "4h"
leverage = 1.0