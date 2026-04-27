#!/usr/bin/env python3
"""
Hypothesis: Weekly Bollinger Band mean reversion with daily volume confirmation.
Trades reversals at weekly Bollinger Band upper/lower bands when daily volume exceeds average,
focusing on mean reversion in ranging markets which performs well in both bull and bear regimes.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""

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
    
    # Get weekly data for Bollinger Bands
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly Bollinger Bands (20, 2)
    weekly_close = df_weekly['close'].values
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Middle band: 20-period SMA
    weekly_ma = pd.Series(weekly_close).rolling(window=20, min_periods=20).mean().values
    # Standard deviation
    weekly_std = pd.Series(weekly_close).rolling(window=20, min_periods=20).std().values
    # Upper and lower bands
    weekly_upper = weekly_ma + 2 * weekly_std
    weekly_lower = weekly_ma - 2 * weekly_std
    
    # Align weekly bands to daily timeframe
    upper_aligned = align_htf_to_ltf(prices, df_weekly, weekly_upper)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, weekly_lower)
    ma_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ma)
    
    # Get daily data for volume filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    # Calculate daily volume average (20-period)
    daily_volume = df_daily['volume'].values
    volume_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_daily, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Bollinger Bands and volume MA
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        volume_now = volume[i]
        vol_ma = volume_ma_aligned[i]
        upper_now = upper_aligned[i]
        lower_now = lower_aligned[i]
        ma_now = ma_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = volume_now > 1.5 * vol_ma
        
        # Entry conditions: Bollinger Band reversal with volume confirmation
        if position == 0:
            # Long: price at or below lower band with volume confirmation
            if price_now <= lower_now and vol_filter:
                signals[i] = size
                position = 1
            # Short: price at or above upper band with volume confirmation
            elif price_now >= upper_now and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or touches upper band
            if price_now >= ma_now or price_now >= upper_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to middle band or touches lower band
            if price_now <= ma_now or price_now <= lower_now:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "weekly_bollinger_mean_reversion_1d_volume"
timeframe = "1d"
leverage = 1.0