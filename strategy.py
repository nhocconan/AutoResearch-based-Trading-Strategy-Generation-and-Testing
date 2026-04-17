#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d volume spike + 1d ADX regime filter.
Long when price breaks above Donchian upper channel with 1d volume > 1.5x 20-period average and 1d ADX > 20.
Short when price breaks below Donchian lower channel with 1d volume > 1.5x 20-period average and 1d ADX > 20.
Exit when price crosses Donchian midline (20-period average of high/low) or ADX drops below 15 (range regime).
Uses 4h for price action and Donchian channels, 1d for volume and ADX regime filters.
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Get 1d data for volume and ADX regime filters
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
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
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(adx_14_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Price levels
        price = close[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        midline = donchian_mid[i]
        
        # 1d regime filters
        adx_val = adx_14_aligned[i]
        vol_ma = vol_ma_20_aligned[i]
        volume_spike = volume_1d[i] > (1.5 * vol_ma) if not np.isnan(vol_ma) and vol_ma > 0 else False
        
        # Trending regime: ADX > 20
        is_trending = adx_val > 20
        
        if position == 0:
            # Long: price breaks above upper channel with volume spike in trending regime
            if price > upper_channel and volume_spike and is_trending:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume spike in trending regime
            elif price < lower_channel and volume_spike and is_trending:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below midline OR ADX drops below 15 (range regime)
            if price < midline or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above midline OR ADX drops below 15 (range regime)
            if price > midline or adx_val < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dVolumeSpike_ADX_Regime"
timeframe = "4h"
leverage = 1.0