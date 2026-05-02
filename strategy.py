#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation
# Uses 12h timeframe for signal generation with Donchian channels from 12h data
# 1d EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 20-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Donchian breakouts capture strong momentum moves, trend filter prevents false breakouts in ranging markets

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) channels from 12h data
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Upper channel: highest high of last 20 periods
    upper_channel = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower channel: lowest low of last 20 periods
    lower_channel = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to 12h timeframe (they change only when new 12h bar forms)
    upper_channel_aligned = align_htf_to_ltf(prices, df_12h, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_12h, lower_channel)
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(upper_channel_aligned[i]) or np.isnan(lower_channel_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above upper Donchian + price > 1d EMA50 + volume confirm
            if close[i] > upper_channel_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian + price < 1d EMA50 + volume confirm
            elif close[i] < lower_channel_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below lower Donchian or reverse signal
            if close[i] < lower_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above upper Donchian or reverse signal
            if close[i] > upper_channel_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals