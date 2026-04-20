#!/usr/bin/env python3
"""
6h_1d_1w_PriceChannelBreakout_Volume
Concept: Use daily and weekly high-low channels for breakout with volume confirmation.
- Daily channel: 10-day high/low for trend context
- Weekly channel: 5-week high/low for major support/resistance
- Breakout when price exceeds weekly channel in direction of daily trend
- Volume confirmation to avoid false breakouts
- Works in bull/bear: channels adapt to volatility, volume filters noise
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_PriceChannelBreakout_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily and weekly data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 15 or len(df_1w) < 6:
        return np.zeros(n)
    
    # === Daily: 10-period high/low for trend ===
    high_10d = pd.Series(df_1d['high']).rolling(window=10, min_periods=10).max().values
    low_10d = pd.Series(df_1d['low']).rolling(window=10, min_periods=10).min().values
    close_1d = df_1d['close'].values
    
    # Daily trend: above/below midpoint of 10-day range
    daily_mid = (high_10d + low_10d) / 2.0
    daily_trend_up = close_1d > daily_mid
    daily_trend_down = close_1d < daily_mid
    
    # === Weekly: 5-period high/low for major S/R ===
    high_5w = pd.Series(df_1w['high']).rolling(window=5, min_periods=5).max().values
    low_5w = pd.Series(df_1w['low']).rolling(window=5, min_periods=5).min().values
    
    # Align all to 6h timeframe
    high_10d_aligned = align_htf_to_ltf(prices, df_1d, high_10d)
    low_10d_aligned = align_htf_to_ltf(prices, df_1d, low_10d)
    daily_trend_up_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_up.astype(float))
    daily_trend_down_aligned = align_htf_to_ltf(prices, df_1d, daily_trend_down.astype(float))
    high_5w_aligned = align_htf_to_ltf(prices, df_1w, high_5w)
    low_5w_aligned = align_htf_to_ltf(prices, df_1w, low_5w)
    
    # === 6h: Price and volume ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # For volume MA
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        vol_ratio_val = vol_ratio[i]
        high_10d_val = high_10d_aligned[i]
        low_10d_val = low_10d_aligned[i]
        daily_trend_up_val = daily_trend_up_aligned[i] > 0.5
        daily_trend_down_val = daily_trend_down_aligned[i] > 0.5
        high_5w_val = high_5w_aligned[i]
        low_5w_val = low_5w_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(vol_ratio_val) or np.isnan(high_10d_val) or 
            np.isnan(low_10d_val) or np.isnan(high_5w_val) or np.isnan(low_5w_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly high with volume and daily uptrend
            breakout_long = close_val > high_5w_val
            vol_confirm = vol_ratio_val > 1.5
            
            if breakout_long and vol_confirm and daily_trend_up_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly low with volume and daily downtrend
            elif close_val < low_5w_val and vol_confirm and daily_trend_down_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to daily midpoint or weekly low
            if close_val <= daily_mid[i] or close_val <= low_5w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns to daily midpoint or weekly high
            if close_val >= daily_mid[i] or close_val >= high_5w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals