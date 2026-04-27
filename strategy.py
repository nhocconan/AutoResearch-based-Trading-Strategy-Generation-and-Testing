#!/usr/bin/env python3
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
    
    # Get daily data for pivot levels and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate weekly EMA(34) for additional trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate Donchian(20) from previous day's daily data
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA, Donchian, volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        ema_trend_1d = ema34_1d_aligned[i]
        ema_trend_1w = ema34_1w_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Only trade when both daily and weekly trends agree (bullish: daily > weekly)
            trend_aligned = (ema_trend_1d > ema_trend_1w)
            
            # Donchian breakout with volume and trend confirmation
            # Long: break above upper band with volume spike and bullish trend
            if (high[i] > high_20_aligned[i] and close[i] > high_20_aligned[i] and 
                trend_aligned and vol_spike_val):
                signals[i] = size
                position = 1
            # Short: break below lower band with volume spike and bearish trend
            elif (low[i] < low_20_aligned[i] and close[i] < low_20_aligned[i] and 
                  not trend_aligned and vol_spike_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian band or trend turns bearish
            if low[i] <= low_20_aligned[i] or not trend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper Donchian band or trend turns bullish
            if high[i] >= high_20_aligned[i] or trend_aligned:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1d1wEMA34_Trend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0