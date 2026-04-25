#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeConfirm_v1
Hypothesis: Daily Donchian(20) breakouts with 1-week EMA50 trend filter and volume confirmation (>1.5x 20-day average volume).
Primary timeframe 1d targets 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
1-week EMA50 provides smooth trend alignment that works in both bull and bear markets.
Volume confirmation ensures breakouts have conviction. Discrete sizing (0.25) manages drawdown.
Designed for BTC/ETH with weekly trend filter to avoid counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian(20) channels from daily data
    # Upper channel: highest high of past 20 days (including current)
    # Lower channel: lowest low of past 20 days (including current)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for volume confirmation
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma20 * 1.5  # 1.5x 20-day average volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian channels and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or
            np.isnan(vol_threshold[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 1.5x 20-day average volume
            vol_confirm = volume[i] > vol_threshold[i]
            
            # Long: price breaks above Donchian upper channel in uptrend (close > 1w EMA50) with volume confirmation
            # Short: price breaks below Donchian lower channel in downtrend (close < 1w EMA50) with volume confirmation
            long_signal = (close[i] > donchian_upper[i]) and (close[i] > ema50_1w_aligned[i]) and vol_confirm
            short_signal = (close[i] < donchian_lower[i]) and (close[i] < ema50_1w_aligned[i]) and vol_confirm
            
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
            # Exit when price moves back below Donchian middle (or trend reversal)
            exit_signal = close[i] < donchian_lower[i]  # Exit if price breaks below lower channel
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian upper channel
            exit_signal = close[i] > donchian_upper[i]  # Exit if price breaks above upper channel
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0