#!/usr/bin/env python3
"""
1d_Donchian20_VolumeSpike_HTFTrend_v1
Hypothesis: On 1d timeframe, take breakout entries when price breaks Donchian(20) channel with volume confirmation (vol_ratio > 1.5) only when aligned with 1w trend (price above/below EMA50 on weekly). Uses discrete sizing (0.25) and ATR-based stoploss. Target: 20-60 trades over 4 years (5-15/year) by requiring tight confluence of Donchian breakout, volume spike, and weekly trend alignment. Designed to work in both bull (breakouts) and bear (breakdowns) markets.
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
    
    # Get 1d and 1w data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d volume ratio (current vs 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    atr_at_entry = 0.0
    
    # Start index: need warmup for all indicators
    start_idx = max(60, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i]):
            # Hold current position or flat if no position
            if position == 0:
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Determine 1w trend (bullish = price above EMA50)
        close_price = close[i]
        htf_1w_bullish = close_price > ema_50_1w_aligned[i]
        htf_1w_bearish = close_price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian upper + volume confirmation + 1w bullish trend
            long_setup = (close_price > high_roll[i]) and (vol_ratio[i] > 1.5) and htf_1w_bullish
            
            # Short setup: price breaks below Donchian lower + volume confirmation + 1w bearish trend
            short_setup = (close_price < low_roll[i]) and (vol_ratio[i] > 1.5) and htf_1w_bearish
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close_price
                atr_at_entry = atr[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close_price
                atr_at_entry = atr[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower OR ATR-based stoploss (2*ATR)
            if (close_price < low_roll[i]) or (close_price < entry_price - 2.0 * atr_at_entry):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper OR ATR-based stoploss (2*ATR)
            if (close_price > high_roll[i]) or (close_price > entry_price + 2.0 * atr_at_entry):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian20_VolumeSpike_HTFTrend_v1"
timeframe = "1d"
leverage = 1.0