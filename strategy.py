#!/usr/bin/env python3
"""
1D_WeeklyVWAP_Retest_4hTrend_Filter
Hypothesis: Weekly VWAP acts as strong support/resistance on daily chart. Price retests weekly VWAP with 4h trend continuation offer high-probability entries. Weekly trend filter (price above/below weekly VWAP) avoids counter-trend trades. Designed for low frequency (10-25 trades/year) to minimize fee drag in 2025 ranging market. Works in both bull (buy VWAP support in uptrend) and bear (sell VWAP resistance in downtrend).
"""

name = "1D_WeeklyVWAP_Retest_4hTrend_Filter"
timeframe = "1d"
leverage = 1.0

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
    
    # Load weekly data ONCE for VWAP and trend filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP: typical price * volume / cumulative volume
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    vwap_num = np.cumsum(typical_price_1w * volume_1w)
    vwap_den = np.cumsum(volume_1w)
    vwap_1w = np.where(vwap_den != 0, vwap_num / vwap_den, typical_price_1w)
    
    # Weekly trend filter: price above/below VWAP
    trend_up_1w = close_1w > vwap_1w
    trend_down_1w = close_1w < vwap_1w
    
    # Align weekly VWAP and trend to daily timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w.astype(float))
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w.astype(float))
    
    # Load 4h data ONCE for trend confirmation
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    # Simple 4h trend: price above/below 20-period EMA
    ema20_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    trend_up_4h = close_4h > ema20_4h
    trend_down_4h = close_4h < ema20_4h
    
    # Align 4h trend to daily timeframe
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h.astype(float))
    trend_down_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_down_4h.astype(float))
    
    # Position size
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(trend_up_1w_aligned[i]) or 
            np.isnan(trend_down_1w_aligned[i]) or np.isnan(trend_up_4h_aligned[i]) or 
            np.isnan(trend_down_4h_aligned[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        near_vwap = abs(close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i] < 0.005  # Within 0.5% of VWAP
        price_above_vwap = close[i] > vwap_1w_aligned[i]
        price_below_vwap = close[i] < vwap_1w_aligned[i]
        
        if position == 0:
            # Long: Price near VWAP from above + weekly uptrend + 4h uptrend
            if near_vwap and price_above_vwap and trend_up_1w_aligned[i] and trend_up_4h_aligned[i]:
                signals[i] = position_size
                position = 1
            # Short: Price near VWAP from below + weekly downtrend + 4h downtrend
            elif near_vwap and price_below_vwap and trend_down_1w_aligned[i] and trend_down_4h_aligned[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions: price moves 1.5% away from VWAP OR trend reverses
            if position == 1:
                vwap_distance = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
                if vwap_distance > 0.015 or not trend_up_1w_aligned[i] or not trend_up_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                vwap_distance = (close[i] - vwap_1w_aligned[i]) / vwap_1w_aligned[i]
                if vwap_distance < -0.015 or not trend_down_1w_aligned[i] or not trend_down_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals