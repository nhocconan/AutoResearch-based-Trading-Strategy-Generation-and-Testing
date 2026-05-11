#!/usr/bin/env python3
"""
1d_1w_WeeklyChannelBreakout_TrendFollow
Hypothesis: Use 1-week Donchian channels as macro trend filter and 1-day Donchian breakouts for entry.
In weekly uptrend (price above weekly 20-period Donchian middle), go long on daily breakouts above upper channel.
In weekly downtrend (price below weekly 20-period Donchian middle), go short on daily breakouts below lower channel.
Exit when price crosses back to weekly Donchian middle or reverses trend.
This captures trend continuation with multi-timeframe confirmation, reducing false signals.
Target: 15-25 trades/year (60-100 total over 4 years) to avoid fee drag.
Works in bull by buying dips in uptrend; works in bear by selling rallies in downtrend.
"""

name = "1d_1w_WeeklyChannelBreakout_TrendFollow"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- Weekly Donchian Channel (20-period) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian upper and lower
    donch_high_1w = np.full(len(high_1w), np.nan)
    donch_low_1w = np.full(len(low_1w), np.nan)
    
    for i in range(20, len(high_1w)):
        donch_high_1w[i] = np.max(high_1w[i-20:i])
        donch_low_1w[i] = np.min(low_1w[i-20:i])
    
    # Weekly Donchian middle (average of upper and lower)
    donch_mid_1w = (donch_high_1w + donch_low_1w) / 2.0
    
    # Align weekly Donchian levels to daily timeframe
    donch_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_high_1w)
    donch_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_low_1w)
    donch_mid_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_mid_1w)
    
    # --- Daily Donchian Breakout (20-period) for entry timing ---
    donch_high_daily = np.full(n, np.nan)
    donch_low_daily = np.full(n, np.nan)
    
    for i in range(20, n):
        donch_high_daily[i] = np.max(high[i-20:i])
        donch_low_daily[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donch_high_1w_aligned[i]) or np.isnan(donch_low_1w_aligned[i]) or 
            np.isnan(donch_mid_1w_aligned[i]) or np.isnan(donch_high_daily[i]) or 
            np.isnan(donch_low_daily[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend based on price relative to weekly Donchian middle
        weekly_uptrend = close[i] > donch_mid_1w_aligned[i]
        weekly_downtrend = close[i] < donch_mid_1w_aligned[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend
            if weekly_uptrend and close[i] > donch_high_daily[i]:
                # Long: weekly uptrend + daily breakout above upper channel
                signals[i] = 0.25
                position = 1
            elif weekly_downtrend and close[i] < donch_low_daily[i]:
                # Short: weekly downtrend + daily breakdown below lower channel
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns down OR price crosses below weekly Donchian middle
                if not weekly_uptrend or close[i] < donch_mid_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR price crosses above weekly Donchian middle
                if not weekly_downtrend or close[i] > donch_mid_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals