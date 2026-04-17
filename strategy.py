#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ADX regime filter + volume confirmation.
Long when price breaks above Donchian(20) high AND 1d ADX > 20 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND 1d ADX > 20 AND volume > 1.5x 20-period average.
Exit when price returns to Donchian midpoint OR ADX < 20 (range regime) OR volume condition fails.
Uses 1d ADX for trend regime filtering to avoid whipsaws in sideways markets.
Target: 75-200 total trades over 4 years (19-50/year) on BTC/ETH/SOL.
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
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period) with proper Wilder smoothing
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
        
        # Wilder's smoothing (alpha = 1/period)
        atr = np.zeros_like(tr)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
        return adx
    
    # Calculate 1d ADX
    adx_14 = calculate_adx(high_1d, low_1d, close_1d, 14)
    adx_14_aligned = align_htf_to_ltf(prices, df_1d, adx_14)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = np.full_like(high, np.nan)
    lowest_low = np.full_like(low, np.nan)
    
    for i in range(lookback-1, len(high)):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate volume average (20-period)
    vol_avg = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(30, 20)  # warmup for Donchian and volume avg
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_avg[i]) or 
            np.isnan(adx_14_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Conditions
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        donchian_mid = (donchian_high + donchian_low) / 2
        vol_ratio = volume[i] / vol_avg[i] if vol_avg[i] > 0 else 0
        vol_confirm = vol_ratio > 1.5
        trend_regime = adx_14_aligned[i] > 20
        
        price = close[i]
        
        if position == 0:
            # Long: break above Donchian high + volume confirmation + trend regime
            if price > donchian_high and vol_confirm and trend_regime:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + volume confirmation + trend regime
            elif price < donchian_low and vol_confirm and trend_regime:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR trend regime ends OR volume confirmation fails
            if price < donchian_mid or not trend_regime or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR trend regime ends OR volume confirmation fails
            if price > donchian_mid or not trend_regime or not vol_confirm:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_DonchianBreakout_1dADX_Volume_Regime"
timeframe = "4h"
leverage = 1.0