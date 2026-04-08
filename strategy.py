#!/usr/bin/env python3
# 12h_donchian_breakout_daily_trend_volume_v1
# Hypothesis: 12h Donchian breakout (20-period) with 1d trend filter (EMA50) and volume confirmation.
# Enters long on breakout above upper band with EMA50-up and volume > 1.5x average; short on breakout below lower band with EMA50-down.
# Exits on reverse breakout or trend failure. Designed for 15-25 trades/year on 12h to minimize fee drag.
# Works in bull/bear via trend-following with volume and trend filters.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_daily_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1-day data for trend filter and Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period) on daily
    highest_high = np.full_like(high_1d, np.nan)
    lowest_low = np.full_like(low_1d, np.nan)
    
    for i in range(20, len(high_1d)):
        highest_high[i] = np.max(high_1d[i-20:i])
        lowest_low[i] = np.min(low_1d[i-20:i])
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Average volume (20-period) for confirmation
    avg_vol_20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        avg_vol_20[i] = np.mean(volume[i-20:i])
    
    # Align all indicators to 12h timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_1d, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1d, lowest_low)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        if np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_vol_20_aligned[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = volume[i] > 1.5 * avg_vol_20_aligned[i]
        price_above_ema = close[i] > ema50_1d_aligned[i]
        price_below_ema = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: break below lower band or trend failure
            if close[i] < lowest_low_aligned[i] or not price_above_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above upper band or trend failure
            if close[i] > highest_high_aligned[i] or not price_below_ema:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above upper band with trend and volume
            if close[i] > highest_high_aligned[i] and price_above_ema and vol_surge:
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower band with trend and volume
            elif close[i] < lowest_low_aligned[i] and price_below_ema and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals