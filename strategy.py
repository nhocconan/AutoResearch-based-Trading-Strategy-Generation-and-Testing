#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_12hTrend_VolumeSpike
Hypothesis: Donchian channel (20-bar) breakouts on 4h with 12h EMA50 trend filter and volume confirmation (>2x 20-bar average).
Breakouts above upper band in uptrend or below lower band in downtrend with volume spike signal institutional participation.
Exits on reversion to the Donchian midpoint (mean reversion). Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull markets (breakouts with trend) and bear markets (breakdowns against trend) by following 12h trend.
Target: 20-50 trades/year to avoid fee drag.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian channels (20-bar) on 4h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Align HTF indicators to 4h timeframe (completed 12h bar lag)
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h, additional_delay_bars=1)
    
    # Volume confirmation: 2.0x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian and EMA
    start_idx = max(50, lookback)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(donchian_mid[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for breakout signals in direction of 12h trend with volume confirmation
            # Long: price breaks above upper Donchian band in uptrend (close > EMA50)
            # Short: price breaks below lower Donchian band in downtrend (close < EMA50)
            long_signal = (close[i] > highest_high[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < lowest_low[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian midpoint (mean reversion)
            exit_signal = close[i] < donchian_mid[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian midpoint (mean reversion)
            exit_signal = close[i] > donchian_mid[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0