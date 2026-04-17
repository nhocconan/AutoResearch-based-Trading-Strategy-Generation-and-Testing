#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d volume spike and ADX regime filter.
Long when price breaks above Donchian(20) high AND 1d volume > 1.5x 20-period average AND (1d ADX < 20 OR price > 1d EMA50).
Short when price breaks below Donchian(20) low AND 1d volume > 1.5x 20-period average AND (1d ADX < 20 OR price < 1d EMA50).
Exit when price reverses to touch Donchian(20) midpoint OR volume drops below average.
Uses 12h for price action, 1d for volume/ADX/EMA filters.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime filters (volume, ADX, EMA)
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
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d volume 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 12h Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    donchian_upper, donchian_lower, donchian_middle = calculate_donchian(high, low, 20)
    
    # Align 1d indicators
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(adx_14_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(donchian_middle[i])):
            signals[i] = 0.0
            continue
        
        # Regime determination
        adx_val = adx_14_aligned[i]
        ema50_val = ema50_1d_aligned[i]
        vol_ma_val = vol_ma_20_aligned[i]
        price = close[i]
        vol = volume[i]
        
        # Volume confirmation: current 12h volume > 1.5x 1d volume MA
        volume_confirm = vol > 1.5 * vol_ma_val
        
        # Regime filter: range (ADX < 20) OR trend (price > EMA50 for long, price < EMA50 for short)
        is_range = adx_val < 20
        is_trend_long = price > ema50_val
        is_trend_short = price < ema50_val
        
        if position == 0:
            # Long: price breaks above Donchian upper AND volume confirm AND (range OR trend long)
            if price > donchian_upper[i] and volume_confirm and (is_range or is_trend_long):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND volume confirm AND (range OR trend short)
            elif price < donchian_lower[i] and volume_confirm and (is_range or is_trend_short):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian middle OR volume drops below average
            if price < donchian_middle[i] or vol < vol_ma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to Donchian middle OR volume drops below average
            if price > donchian_middle[i] or vol < vol_ma_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dVolumeSpike_ADXEMA_Regime"
timeframe = "12h"
leverage = 1.0