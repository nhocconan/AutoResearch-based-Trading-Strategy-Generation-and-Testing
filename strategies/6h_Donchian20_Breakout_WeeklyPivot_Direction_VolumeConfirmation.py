#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
Long when price breaks above 20-period high in weekly bullish bias with volume spike.
Short when price breaks below 20-period low in weekly bearish bias with volume spike.
Weekly pivot direction (price vs weekly VWAP) provides multi-day trend filter to avoid counter-trend trades.
Volume spike confirms institutional interest. Works in bull/bear by following weekly bias.
Discrete position sizing (0.25) minimizes fee churn. Targets 12-37 trades/year on 6h.
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
    
    # Get 1d data for weekly aggregation (need daily to build weekly)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate weekly VWAP from daily data
    # Typical price = (H+L+C)/3
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3.0
    # VWAP = sum(typical_price * volume) / sum(volume) over weekly window
    vol_1d = df_1d['volume'].values
    tp_vol_1d = typical_price_1d.values * vol_1d
    
    # Weekly sums (7-day window)
    tp_vol_sum = pd.Series(tp_vol_1d).rolling(window=7, min_periods=7).sum().values
    vol_sum = pd.Series(vol_1d).rolling(window=7, min_periods=7).sum().values
    weekly_vwap = tp_vol_sum / vol_sum
    weekly_vwap = np.where(vol_sum == 0, np.nan, weekly_vwap)
    
    # Weekly bias: price above/below weekly VWAP
    weekly_bullish = df_1d['close'].values > weekly_vwap
    weekly_bearish = df_1d['close'].values < weekly_vwap
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bullish.astype(float))
    weekly_bearish_aligned = align_htf_to_ltf(prices, df_1d, weekly_bearish.astype(float))
    
    # Get 6h data for Donchian channels
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Calculate Donchian(20) on 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to primary timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian, 20 for volume MA, 7 for weekly VWAP)
    start_idx = max(20, 20, 7)
    
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
            # Long: price breaks above Donchian high with weekly bullish bias and volume spike
            if (close[i] > donchian_high_aligned[i] and 
                weekly_bullish_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with weekly bearish bias and volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  weekly_bearish_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below Donchian low OR weekly bias turns bearish
            if (close[i] < donchian_low_aligned[i] or not weekly_bullish_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above Donchian high OR weekly bias turns bullish
            if (close[i] > donchian_high_aligned[i] or not weekly_bearish_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyPivot_Direction_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0