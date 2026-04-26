#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_VolumeConfirmation
Hypothesis: 1d Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-day high in weekly bullish trend with volume spike.
Short when price breaks below 20-day low in weekly bearish trend with volume spike.
Weekly trend filter (price above/below weekly EMA34) avoids counter-trend trades.
Volume spike confirms institutional interest. Works in bull/bear by following weekly trend.
Discrete position sizing (0.25) minimizes fee churn. Targets 7-25 trades/year on 1d.
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
    
    # Get weekly data for trend filter and Donchian channels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_bullish = close_1w > ema_34_1w
    weekly_bearish = close_1w < ema_34_1w
    
    # Align weekly trend to daily timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bearish.astype(float))
    
    # Calculate weekly Donchian(20) on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Volume confirmation: volume > 2.0x 20-period MA on daily data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for EMA, 20 for Donchian, 20 for volume MA)
    start_idx = max(34, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(weekly_bearish_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with weekly bullish trend and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                weekly_bullish_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with weekly bearish trend and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_bearish_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below weekly Donchian low OR weekly trend turns bearish
            if (close[i] < donchian_low_aligned[i] or not weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above weekly Donchian high OR weekly trend turns bullish
            if (close[i] > donchian_high_aligned[i] or not weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0