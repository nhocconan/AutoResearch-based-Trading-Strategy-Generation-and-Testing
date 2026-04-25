#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_TrendFilter_v1
Hypothesis: Trade Donchian(20) breakouts on 4h timeframe with volume spike confirmation and 1d EMA50 trend filter.
In bull markets: buy when price breaks above upper Donchian(20) + volume > 1.5x MA20 + price > 1d EMA50.
In bear markets: sell when price breaks below lower Donchian(20) + volume > 1.5x MA20 + price < 1d EMA50.
Exit on opposite Donchian touch or trend reversal.
Position size: 0.25 to limit drawdown.
Target: 20-40 trades/year to stay well under 400-trade 4h hard max.
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
    if len(df_1d) < 50:  # Need enough for EMA50
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian(20) on 4h
    lookback = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        upper[i] = np.max(high[i - lookback + 1:i + 1])
        lower[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Calculate volume MA(20) for spike detection
    vol_ma = np.full(n, np.nan)
    vol_lookback = 20
    for i in range(vol_lookback - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_lookback + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and volume MA(20)
    start_idx = max(lookback, vol_lookback) - 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Volume spike condition: current volume > 1.5x MA20
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long setup: price breaks above upper Donchian + volume spike + 1d uptrend
            long_setup = (close[i] > upper[i]) and volume_spike and htf_1d_bullish
            
            # Short setup: price breaks below lower Donchian + volume spike + 1d downtrend
            short_setup = (close[i] < lower[i]) and volume_spike and htf_1d_bearish
            
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
            # Exit: price touches lower Donchian (stop) OR 1d trend turns bearish
            if (close[i] <= lower[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches upper Donchian (stop) OR 1d trend turns bullish
            if (close[i] >= upper[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0