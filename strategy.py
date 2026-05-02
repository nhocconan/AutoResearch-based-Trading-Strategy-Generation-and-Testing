#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Donchian breakouts capture strong momentum moves, effective in both bull and bear markets
# 1d EMA50 provides robust higher-timeframe trend alignment to avoid counter-trend trades
# Volume spike (>2.0 x 20-period EMA) confirms breakout validity while reducing false signals
# Uses discrete position sizing (0.25) to minimize fee churn and control drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee drag

name = "12h_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation (volume spike > 2.0 x 20-period EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (2.0 * vol_ema_20)
    
    # 1d data for Donchian calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous 1d bar (20-period)
    donchian_period = 20
    if len(df_1d) < donchian_period:
        return np.zeros(n)
    
    # Upper channel: highest high over past donchian_period days (excluding current)
    upper_channel = pd.Series(df_1d['high'].values).rolling(window=donchian_period, min_periods=donchian_period).max().shift(1).values
    # Lower channel: lowest low over past donchian_period days (excluding current)
    lower_channel = pd.Series(df_1d['low'].values).rolling(window=donchian_period, min_periods=donchian_period).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe (wait for completed 1d bar)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA and Donchian calculation)
    start_idx = max(50, donchian_period)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_channel_aligned[i]) or 
            np.isnan(lower_channel_aligned[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Close breaks above upper Donchian channel with volume confirmation and uptrend
            if close[i] > upper_channel_aligned[i] and volume_confirmation[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Close breaks below lower Donchian channel with volume confirmation and downtrend
            elif close[i] < lower_channel_aligned[i] and volume_confirmation[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Close drops below lower Donchian channel (reversal) OR trend changes to downtrend
            if close[i] < lower_channel_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Close rises above upper Donchian channel (reversal) OR trend changes to uptrend
            if close[i] > upper_channel_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals