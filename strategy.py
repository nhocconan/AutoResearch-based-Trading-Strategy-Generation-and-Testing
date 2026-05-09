#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d volume spike and 1d EMA trend filter.
# Uses 12h Donchian (20) for breakout signals, confirmed by 1d volume > 2x EMA20 and price > 1d EMA50 for longs,
# or price < 1d EMA50 for shorts. Designed to capture medium-term trends with low trade frequency.
# Works in bull markets via breakouts and in bear via breakdowns with trend filter.
name = "12h_Donchian20_1dVolumeSpike_1dEMA50_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for volume and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h Donchian channel (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # 1d volume EMA20 for spike detection
    vol_ema20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema20_aligned = align_htf_to_ltf(prices, df_1d, vol_ema20)
    
    # 1d EMA50 trend filter
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ema20_aligned[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price > 12h Donchian high + volume spike + price > 1d EMA50
            if (price > donchian_high[i] and 
                volume[i] > (2.0 * vol_ema20_aligned[i]) and 
                price > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < 12h Donchian low + volume spike + price < 1d EMA50
            elif (price < donchian_low[i] and 
                  volume[i] > (2.0 * vol_ema20_aligned[i]) and 
                  price < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < 12h Donchian low or price < 1d EMA50
            if price < donchian_low[i] or price < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > 12h Donchian high or price > 1d EMA50
            if price > donchian_high[i] or price > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals