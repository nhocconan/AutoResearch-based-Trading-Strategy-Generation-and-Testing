#!/usr/bin/env python3
# 6h_1w_Supertrend_WeeklyTrend_Filter_V1
# Hypothesis: On 6h timeframe, use weekly Supertrend (ATR=10, mult=3) to filter long/short bias,
# and enter on 6h Donchian(20) breakout in the direction of the weekly trend with volume confirmation.
# Weekly trend filter reduces whipsaw in sideways/ bear markets, capturing strong moves.
# Target: 15-35 trades per year per symbol to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_Supertrend_WeeklyTrend_Filter_V1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly Supertrend (ATR=10, multiplier=3) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # ATR(10)
    atr_1w = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    basic_ub = (high_1w + low_1w) / 2 + 3 * atr_1w
    basic_lb = (high_1w + low_1w) / 2 - 3 * atr_1w
    
    # Final Upper and Lower Bands
    final_ub = np.zeros_like(basic_ub)
    final_lb = np.zeros_like(basic_lb)
    for i in range(len(close_1w)):
        if i == 0:
            final_ub[i] = basic_ub[i]
            final_lb[i] = basic_lb[i]
        else:
            if basic_ub[i] < final_ub[i-1] or close_1w[i-1] > final_ub[i-1]:
                final_ub[i] = basic_ub[i]
            else:
                final_ub[i] = final_ub[i-1]
            
            if basic_lb[i] > final_lb[i-1] or close_1w[i-1] < final_lb[i-1]:
                final_lb[i] = basic_lb[i]
            else:
                final_lb[i] = final_lb[i-1]
    
    # Supertrend
    supertrend_1w = np.zeros_like(close_1w)
    for i in range(len(close_1w)):
        if i == 0:
            supertrend_1w[i] = final_ub[i]
        else:
            if supertrend_1w[i-1] == final_ub[i-1]:
                if close_1w[i] <= final_ub[i]:
                    supertrend_1w[i] = final_ub[i]
                else:
                    supertrend_1w[i] = final_lb[i]
            else:
                if close_1w[i] >= final_lb[i]:
                    supertrend_1w[i] = final_lb[i]
                else:
                    supertrend_1w[i] = final_ub[i]
    
    # Trend direction: 1 for uptrend (price > Supertrend), -1 for downtrend
    trend_1w = np.where(close_1w > supertrend_1w, 1, -1)
    
    # Align to 6h timeframe
    trend_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 6h: Donchian(20) channels ===
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # === 6h: Volume ratio (current vs 20-period average) ===
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        trend_val = trend_aligned[i]
        dh_val = donchian_high[i]
        dl_val = donchian_low[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(trend_val) or np.isnan(dh_val) or np.isnan(dl_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with weekly uptrend and volume confirmation
            if close_val > dh_val and trend_val == 1 and vol_ratio_val > 2.0:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with weekly downtrend and volume confirmation
            elif close_val < dl_val and trend_val == -1 and vol_ratio_val > 2.0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below Donchian low or weekly trend turns down
            if close_val < dl_val or trend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above Donchian high or weekly trend turns up
            if close_val > dh_val or trend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals