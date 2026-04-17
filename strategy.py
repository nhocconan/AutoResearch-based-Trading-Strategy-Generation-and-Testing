#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and ATR-based trend filter.
Long when price breaks above upper Donchian channel with volume > 1.5x average and ATR(14) > ATR(50) (trending up).
Short when price breaks below lower Donchian channel with volume > 1.5x average and ATR(14) < ATR(50) (trending down).
Exit when price reverts to the midpoint of the Donchian channel.
Uses 1d EMA(50) as higher timeframe trend filter: only long when price > EMA50, only short when price < EMA50.
Target: 75-200 total trades over 4 years (19-50/year). Uses tight entry conditions to avoid overtrading and fee drag.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    mid_channel = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        upper_channel[i] = np.max(high[i - lookback + 1:i + 1])
        lower_channel[i] = np.min(low[i - lookback + 1:i + 1])
        mid_channel[i] = (upper_channel[i] + lower_channel[i]) / 2.0
    
    # Calculate ATR(14) and ATR(50) for trend filter
    def calculate_atr(high, low, close, period):
        atr = np.full_like(close, np.nan)
        tr = np.full_like(close, np.nan)
        
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's ATR
        if len(tr) >= period + 1:
            atr[period] = np.nanmean(tr[1:period+1])
            for i in range(period + 1, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    
    # Calculate volume spike (current volume > 1.5x 20-period average)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(50, lookback)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(mid_channel[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(atr_14[i]) or 
            np.isnan(atr_50[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema_50 = ema_50_1d_aligned[i]
        atr14 = atr_14[i]
        atr50 = atr_50[i]
        upper = upper_channel[i]
        lower = lower_channel[i]
        mid = mid_channel[i]
        
        # Trend filter: ATR(14) > ATR(50) = strengthening trend up, ATR(14) < ATR(50) = strengthening trend down
        trending_up = atr14 > atr50
        trending_down = atr14 < atr50
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume spike and EMA50 filter in uptrend
            if price > upper and vol_spike and trending_up and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume spike and EMA50 filter in downtrend
            elif price < lower and vol_spike and trending_down and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint of Donchian channel
            if price <= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint of Donchian channel
            if price >= mid:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_VolumeSpike_ATRTrend_EMA50Filter"
timeframe = "4h"
leverage = 1.0