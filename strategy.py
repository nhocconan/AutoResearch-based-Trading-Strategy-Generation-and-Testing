#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter
Hypothesis: Donchian(20) breakouts aligned with 1d EMA50 trend, volume confirmation (>2x 24-bar average), and choppiness regime (chop < 38.2) capture sustained moves while avoiding whipsaws in ranging markets. Discrete sizing (0.25) limits fee churn. Target: 75-200 total trades over 4 years (19-50/year).
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) channels: 20-period high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 24-period average (6h equivalent)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    # Choppiness regime filter (14-period) - chop > 61.8 = ranging, chop < 38.2 = trending
    # We only trade in trending regimes (chop < 38.2) to avoid whipsaws
    hl_range = pd.Series(high - low).rolling(window=14, min_periods=14).sum().values
    true_range = pd.Series(np.maximum(high - low, 
                                     np.maximum(np.abs(high - np.append([np.nan], close[:-1])),
                                                np.abs(low - np.append([np.nan], close[:-1]))))).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(hl_range / true_range) / np.log10(14)
    chop_filter = chop < 38.2  # Only trade when market is trending
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need Donchian (20), volume avg (24), chop (14), 1d EMA50 (50)
    start_idx = max(20, 24, 14, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(chop_filter[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_1d_val = ema_50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        is_trending = chop_filter[i]
        
        # Only enter trades in trending regime
        if not is_trending:
            # In ranging markets, exit any position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine trend: price above/below 1d EMA50
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above Donchian high and volume confirms
                if (close_val > upper) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below Donchian low and volume confirms
                if (close_val < lower) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches Donchian low or trend changes to downtrend
            exit_condition = (close_val < lower) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches Donchian high or trend changes to uptrend
            exit_condition = (close_val > upper) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0