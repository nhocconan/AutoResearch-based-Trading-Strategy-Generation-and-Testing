#!/usr/bin/env python3
"""
6h_1W_200WeekEMA_TrendFollowing_V1
Hypothesis: The 200-week EMA acts as a strong long-term trend filter for Bitcoin and Ethereum. In bull markets (price > 200-week EMA), we look for long entries on 6x pullbacks to the 20-week EMA with volume confirmation. In bear markets (price < 200-week EMA), we look for short entries on 6x bounces to the 20-week EMA with volume confirmation. This multi-timeframe approach uses the weekly trend to filter 6-hour entries, reducing false signals and capturing major trends while avoiding counter-trend trades. Designed for low trade frequency (target: 12-37/year) to minimize fee drag in 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for 200-week EMA trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 200:
        return np.zeros(n)
    
    # Calculate weekly 200 EMA
    close_weekly = df_weekly['close'].values
    ema_200_weekly = pd.Series(close_weekly).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly 200 EMA to 6h timeframe
    ema_200_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_200_weekly)
    
    # Load daily data for 20 EMA (used for entry timing on 6x chart)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily 20 EMA
    close_daily = df_daily['close'].values
    ema_20_daily = pd.Series(close_daily).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align daily 20 EMA to 6h timeframe
    ema_20_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_20_daily)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current volume > 1.5x 24-period average (24*6h = 6 days)
    volume_avg = np.zeros_like(volume)
    for i in range(len(volume)):
        if i >= 24:
            volume_avg[i] = np.mean(volume[i-24:i])
        else:
            volume_avg[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
    volume_filter = volume > (1.5 * volume_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(24, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_200_weekly_aligned[i]) or np.isnan(ema_20_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_200w = ema_200_weekly_aligned[i]
        ema_20d = ema_20_daily_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Bull market: price above 200-week EMA -> look for longs on pullbacks to 20-day EMA
            if price > ema_200w:
                # Long: price pulls back to or slightly below 20-day EMA with volume
                if price <= ema_20d * 1.01 and price >= ema_20d * 0.99 and vol_ok:
                    # Additional confirmation: price closing in upper half of range
                    if close[i] > (high[i] + low[i]) / 2:
                        signals[i] = 0.25
                        position = 1
            # Bear market: price below 200-week EMA -> look for shorts on bounces to 20-day EMA
            elif price < ema_200w:
                # Short: price bounces to or slightly above 20-day EMA with volume
                if price >= ema_20d * 0.99 and price <= ema_20d * 1.01 and vol_ok:
                    # Additional confirmation: price closing in lower half of range
                    if close[i] < (high[i] + low[i]) / 2:
                        signals[i] = -0.25
                        position = -1
        
        elif position == 1:
            # Long exit: price crosses below 20-day EMA or re-tests 200-week EMA as resistance
            if price < ema_20d * 0.99 or price > ema_200w * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 20-day EMA or re-tests 200-week EMA as support
            if price > ema_20d * 1.01 or price < ema_200w * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1W_200WeekEMA_TrendFollowing_V1"
timeframe = "6h"
leverage = 1.0