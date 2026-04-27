#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1
Hypothesis: 6h Donchian(20) breakouts aligned with weekly pivot direction (price vs weekly VWAP) and 1d EMA50 trend filter capture high-probability momentum moves. Weekly VWAP acts as dynamic support/resistance. Discrete sizing (0.25) balances return and fee drag. Target: 50-150 total trades over 4 years.
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly VWAP (trend filter)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_num = pd.Series(typical_price_1w * volume_1w).cumsum().values
    vwap_den = pd.Series(volume_1w).cumsum().values
    vwap_1w = np.where(vwap_den != 0, vwap_num / vwap_den, 0.0)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Align all indicators to primary timeframe (6h)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)  # Use 1d as proxy for alignment
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need EMA50 (50), VWAP (need volume), Donchian (20), volume avg (20)
    start_idx = max(50, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vwap_1w_aligned[i]) or 
            np.isnan(volume_confirm_aligned[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema50 = ema50_1d_aligned[i]
        vwap = vwap_1w_aligned[i]
        vol_conf = volume_confirm_aligned[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        
        if position == 0:
            # Determine trend: price above both EMA50 and weekly VWAP = uptrend
            uptrend = close_val > ema50 and close_val > vwap
            downtrend = close_val < ema50 and close_val < vwap
            
            if uptrend and vol_conf:
                # Long when price breaks above Donchian high with volume
                if close_val > upper:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
            elif downtrend and vol_conf:
                # Short when price breaks below Donchian low with volume
                if close_val < lower:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Exit: price re-enters Donchian channel or weekly VWAP
            if close_val < upper and close_val > lower:
                signals[i] = 0.0
                position = 0
            elif close_val < vwap:  # Weekly VWAP break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit: price re-enters Donchian channel or weekly VWAP
            if close_val < upper and close_val > lower:
                signals[i] = 0.0
                position = 0
            elif close_val > vwap:  # Weekly VWAP break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0