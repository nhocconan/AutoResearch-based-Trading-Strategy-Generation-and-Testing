#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dTrend_VolumeSpike
Hypothesis: Donchian(20) breakouts on 6h chart with 1d trend filter (price > EMA50 for longs, < EMA50 for shorts) and volume confirmation (>2.0x average) capture strong momentum moves. Works in bull markets via upside breakouts and bear markets via downside breakdowns. Volume spike ensures institutional participation, reducing false breakouts. Targets 80-160 total trades over 4 years (20-40/year). Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 6h: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need Donchian (20), EMA50 (50), volume avg (20)
    start_idx = max(20, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_50_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above Donchian high and volume confirms
                if (close_val > upper_band) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below Donchian low and volume confirms
                if (close_val < lower_band) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price retouches Donchian midpoint or trend changes to downtrend
            midpoint = (upper_band + lower_band) / 2.0
            exit_condition = (close_val < midpoint) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price retouches Donchian midpoint or trend changes to uptrend
            midpoint = (upper_band + lower_band) / 2.0
            exit_condition = (close_val > midpoint) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0