#!/usr/bin/env python3
"""
4h_1D_Donchian20_Breakout_Volume_Confirmation_v3
Hypothesis: Buy when price breaks above 4h Donchian(20) high with volume > 1.5x average, sell when breaks below low. Uses daily trend filter (price > daily EMA50) to avoid counter-trend trades. Designed for low trade frequency (target: 20-50/year) to minimize fee drag. Works in bull markets via breakouts and in bear markets via short breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data once for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = pd.Series(close_daily).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Main timeframe data (4h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_avg = np.full(n, np.nan)
    for i in range(20, n):
        volume_avg[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if NaN in critical values
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema50 = ema50_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and uptrend filter
            if price > upper and vol_ok and price > ema50:
                signals[i] = 0.30
                position = 1
            # Short: break below Donchian low with volume and downtrend filter
            elif price < lower and vol_ok and price < ema50:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reverses
            if price < lower or price < ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reverses
            if price > upper or price > ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_1D_Donchian20_Breakout_Volume_Confirmation_v3"
timeframe = "4h"
leverage = 1.0