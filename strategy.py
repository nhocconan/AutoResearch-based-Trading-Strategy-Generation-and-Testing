#!/usr/bin/env python3
"""
1d_Weekly_Donchian_Breakout_Trend_Filter
Hypothesis: Weekly Donchian channel breakout with daily trend filter and volume confirmation.
In bull markets (price > daily EMA50), long on weekly upper band breakout with volume.
In bear markets (price < daily EMA50), short on weekly lower band breakout with volume.
Uses weekly Donchian channels for structure and daily EMA for trend filter to avoid counter-trend trades.
Target: 15-25 trades per year (~60-100 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Donchian_Breakout_Trend_Filter"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels: 20-period high/low
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Donchian channels
    upper_channel = np.full_like(weekly_high, np.nan, dtype=np.float64)
    lower_channel = np.full_like(weekly_low, np.nan, dtype=np.float64)
    
    for i in range(20, len(weekly_high)):
        upper_channel[i] = np.max(weekly_high[i-20:i])
        lower_channel[i] = np.min(weekly_low[i-20:i])
    
    # Align weekly channels to daily timeframe
    upper_channel_aligned = align_htf_to_ltf(prices, df_weekly, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_weekly, lower_channel)
    
    # Daily EMA50 for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume ratio: current volume / 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need 50 periods for EMA50 and sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_50[i]) or np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market regime from daily EMA50
        uptrend_regime = close[i] > ema_50[i]
        downtrend_regime = close[i] < ema_50[i]
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: close breaks above weekly upper channel in uptrend regime + volume
            long_entry = (close[i] > upper_channel_aligned[i]) and uptrend_regime and volume_confirm
            # Short: close breaks below weekly lower channel in downtrend regime + volume
            short_entry = (close[i] < lower_channel_aligned[i]) and downtrend_regime and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: close crosses below weekly lower channel or regime changes to downtrend
            if (close[i] < lower_channel_aligned[i]) or (not uptrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: close crosses above weekly upper channel or regime changes to uptrend
            if (close[i] > upper_channel_aligned[i]) or (not downtrend_regime):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals