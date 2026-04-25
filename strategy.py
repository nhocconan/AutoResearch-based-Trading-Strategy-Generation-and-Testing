#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1
Hypothesis: Trade Donchian(20) breakouts on 12h timeframe with 1d EMA50 trend filter and volume spike confirmation.
In bull markets: buy when price breaks above 20-bar high AND price > 1d EMA50 AND volume > 1.5x 20-bar average volume.
In bear markets: sell when price breaks below 20-bar low AND price < 1d EMA50 AND volume > 1.5x 20-bar average volume.
Exit on opposite Donchian breakout or trend reversal.
Position size: 0.25 to limit drawdown in volatile markets.
Target: 15-30 trades/year to stay well under 200-trade 12h hard max.
Works in bull (breakouts with uptrend) and bear (breakdowns with downtrend) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need at least 50 bars for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 20-bar Donchian channels and volume average
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    vol_roll = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 20-bar calculations
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_roll[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume spike condition: current volume > 1.5x 20-bar average volume
        volume_spike = volume[i] > 1.5 * vol_roll[i]
        
        if position == 0:
            # Long setup: price breaks above 20-bar high + 1d uptrend + volume spike
            long_setup = (close[i] > high_roll[i]) and htf_1d_bullish and volume_spike
            
            # Short setup: price breaks below 20-bar low + 1d downtrend + volume spike
            short_setup = (close[i] < low_roll[i]) and htf_1d_bearish and volume_spike
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below 20-bar low OR 1d trend turns bearish
            if (close[i] < low_roll[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above 20-bar high OR 1d trend turns bullish
            if (close[i] > high_roll[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0