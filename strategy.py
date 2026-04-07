#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d volume confirmation and 1w trend filter
# Uses Donchian channel (20) for breakout direction, 1d volume surge for confirmation,
# and 1w EMA trend filter to avoid counter-trend trades.
# Designed for low trade frequency (target: 12-37 trades/year) to minimize fee drag.
# Works in bull markets via trend-following breakouts and in bear markets via mean-reversion
# at channel extremes when counter-trend volume spikes occur.

name = "12h_donchian20_1d_volume_1w_ema_filter_v1"
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
    
    # 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Donchian Channel (20) - using 20-period lookback
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(20, n):
        highest_high[i] = np.max(high[i-20:i])
        lowest_low[i] = np.min(low[i-20:i])
    
    # 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    # 1w EMA (50) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(50, len(close_1w)):
        ema_1w[i] = close_1w[i] * (2 / (50 + 1)) + ema_1w[i-1] * (1 - 2 / (50 + 1))
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_vol_1d_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x average
        vol_1d_idx = i // 2  # 12h to 1d approximation (2 bars per day)
        if vol_1d_idx >= len(df_1d):
            vol_1d_idx = len(df_1d) - 1
        vol_1d_current = df_1d['volume'].iloc[vol_1d_idx] if vol_1d_idx < len(df_1d) else 0
        vol_surge = vol_1d_current > (1.5 * avg_vol_1d_aligned[i]) if avg_vol_1d_aligned[i] > 0 else False
        
        # Trend filter: price above/below 1w EMA
        price_above_ema = close[i] > ema_1w_aligned[i]
        price_below_ema = close[i] < ema_1w_aligned[i]
        
        # Breakout conditions
        breakout_up = close[i] > highest_high[i]
        breakout_down = close[i] < lowest_low[i]
        
        # Long: upward breakout with volume surge and uptrend filter
        if breakout_up and vol_surge and price_above_ema:
            signals[i] = 0.25
        # Short: downward breakout with volume surge and downtrend filter
        elif breakout_down and vol_surge and price_below_ema:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals