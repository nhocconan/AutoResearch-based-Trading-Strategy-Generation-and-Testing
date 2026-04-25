#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dATR_VolumeSpike_Trend
Hypothesis: Trade Donchian(20) breakouts on 4h with 1d ATR-based volatility filter and volume spike confirmation. Uses discrete sizing (0.25) to limit fee drag. Trend filter uses 1d EMA34 to avoid counter-trend trades in bear markets. Designed for both bull and bear markets by aligning with 1d trend. Target: 20-40 trades/year per symbol.
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
    
    # Get 1d data for HTF trend filter, ATR, and Donchian reference (using 1d for stability)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]  # avoid NaN on first
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian(20) channels from 4h data (using lookback of 20 periods)
    # We need at least 20 bars of lookback, so we calculate manually
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian (20), EMA34 (34), ATR (14), volume MA (20)
    start_idx = max(20, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA34)
        htf_1d_bullish = close[i] > ema_34_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_34_1d_aligned[i]
        
        # Volatility filter: only trade when ATR is above its 50-period MA (avoid low volatility choppy periods)
        atr_ma_50 = pd.Series(atr_14_1d_aligned).rolling(window=50, min_periods=50).mean().values
        volatility_filter = not np.isnan(atr_ma_50[i]) and atr_14_1d_aligned[i] > atr_ma_50[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + 1d uptrend + volume confirmation + volatility filter
            long_setup = (close[i] > highest_20[i]) and htf_1d_bullish and volume_confirm[i] and volatility_filter
            
            # Short setup: price breaks below Donchian lower + 1d downtrend + volume confirmation + volatility filter
            short_setup = (close[i] < lowest_20[i]) and htf_1d_bearish and volume_confirm[i] and volatility_filter
            
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
            # Exit: price touches Donchian lower (stop) OR 1d trend turns bearish
            if (close[i] <= lowest_20[i]) or (not htf_1d_bullish):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price touches Donchian upper (stop) OR 1d trend turns bullish
            if (close[i] >= highest_20[i]) or (htf_1d_bullish):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_1dATR_VolumeSpike_Trend"
timeframe = "4h"
leverage = 1.0