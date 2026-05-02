#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# Uses 12h timeframe for signal generation with Donchian channels from 20-period lookback
# 1w EMA50 provides higher timeframe trend filter to avoid counter-trend trades
# Volume confirmation (1.8x 30-period average) ensures institutional participation
# Discrete position sizing (0.25) balances return and risk
# Target: 50-150 total trades over 4 years = 12-37/year for 12h timeframe
# Works in bull markets via trend-aligned breakouts, in bear via trend filter avoiding false signals
# Based on proven Donchian breakout structure with volume confirmation edge

name = "12h_Donchian20_1wEMA50_Trend_VolumeConfirm_v1"
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
    
    # Load 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels (20-period) on 12h data
    # Use rolling window with min_periods to avoid look-ahead
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation (1.8x 30-period average)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i]) or
            np.isnan(high_roll[i]) or np.isnan(low_roll[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper + price > 1w EMA50 + volume confirm
            if close[i] > high_roll[i] and close[i] > ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower + price < 1w EMA50 + volume confirm
            elif close[i] < low_roll[i] and close[i] < ema_50_1w_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower or reverse signal
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper or reverse signal
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals