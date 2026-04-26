#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeATRStop
Hypothesis: Donchian(20) breakout with 1d trend filter and volume confirmation captures strong momentum moves in both bull and bear markets. 
In bull markets: price breaks above upper Donchian channel with 1d uptrend and volume spike → long. 
In bear markets: price breaks below lower Donchian channel with 1d downtrend and volume spike → short. 
Uses ATR-based trailing stoploss to limit drawdown. Target: 75-200 trades over 4 years. 
Donchian channels provide objective breakout levels that work across regimes and timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:  # Need 20 for Donchian + warmup
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    upper_channel = high_roll.max().values
    lower_channel = low_roll.min().values
    
    # ATR for stoploss and position sizing adjustment
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 20 for Donchian, 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or 
            np.isnan(volume_spike[i]) or 
            np.isnan(atr[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        ema_val = ema_50_1d_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        atr_val = atr[i]
        
        # Breakout conditions
        long_breakout = close_val > upper_val
        short_breakout = close_val < lower_val
        
        # Entry logic: breakout with volume spike and trend alignment
        long_entry = long_breakout and volume_spike[i] and (close_val > ema_val)
        short_entry = short_breakout and volume_spike[i] and (close_val < ema_val)
        
        # Exit logic: ATR-based trailing stop
        exit_long = False
        exit_short = False
        
        if position == 1:  # Long position
            # Calculate highest high since entry (approximate with rolling max of recent bars)
            lookback = min(bars_since_entry, 50)  # Limit lookback for performance
            if lookback > 0:
                recent_high = np.max(high[i-lookback:i+1])
                trailing_stop = recent_high - (2.5 * atr_val)
                exit_long = close_val < trailing_stop
        elif position == -1:  # Short position
            # Calculate lowest low since entry
            lookback = min(bars_since_entry, 50)
            if lookback > 0:
                recent_low = np.min(low[i-lookback:i+1])
                trailing_stop = recent_low + (2.5 * atr_val)
                exit_short = close_val > trailing_stop
        
        # Minimum holding period: 1 bar to reduce churn
        if position != 0 and bars_since_entry < 1:
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_entry and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_entry and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeATRStop"
timeframe = "4h"
leverage = 1.0