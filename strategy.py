#!/usr/bin/env python3
"""
1d_WeeklyDonchianBreakout_VolumeConfirm_TrendFilter_v1
Hypothesis: On 1d timeframe, enter long when price breaks above weekly Donchian(20) high with volume > 1.5x 20-day average volume AND weekly trend is up (close > weekly EMA34); enter short when price breaks below weekly Donchian(20) low with volume confirmation AND weekly trend is down. Exit on opposite Donchian break or trend reversal. Uses weekly structure for direction, daily for execution, volume confirmation to avoid false breakouts, and discrete sizing (0.0, ±0.25) to minimize fee churn. Designed to work in bull via trend continuation and in bear via mean-reversion at extremes with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:  # need for EMA34 and Donchian(20)
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Donchian high: 20-period rolling max
    donchian_high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Donchian low: 20-period rolling min
    donchian_low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20)
    
    # Weekly EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Daily volume confirmation: volume > 1.5x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian(20), EMA34, and volume MA warmup
    start_idx = max(20, 34, 20)  # 34 is the highest
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_20_aligned[i]) or 
            np.isnan(donchian_low_20_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Weekly trend filter
        weekly_uptrend = close_1w_aligned[i] > ema_34_1w_aligned[i]  # need weekly close aligned
        weekly_downtrend = close_1w_aligned[i] < ema_34_1w_aligned[i]
        
        # Get aligned weekly close for trend comparison
        df_1w_close = get_htf_data(prices, '1w')['close'].values
        close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w_close)
        
        weekly_uptrend = close_1w_aligned[i] > ema_34_1w_aligned[i]
        weekly_downtrend = close_1w_aligned[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + volume confirmed + weekly uptrend
            long_breakout = close[i] > donchian_high_20_aligned[i]
            long_signal = long_breakout and volume_confirmed[i] and weekly_uptrend
            
            # Short: price breaks below weekly Donchian low + volume confirmed + weekly downtrend
            short_breakout = close[i] < donchian_low_20_aligned[i]
            short_signal = short_breakout and volume_confirmed[i] and weekly_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below weekly Donchian low OR weekly trend turns down
            if close[i] < donchian_low_20_aligned[i] or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above weekly Donchian high OR weekly trend turns up
            if close[i] > donchian_high_20_aligned[i] or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchianBreakout_VolumeConfirm_TrendFilter_v1"
timeframe = "1d"
leverage = 1.0