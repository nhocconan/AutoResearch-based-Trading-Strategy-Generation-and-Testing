#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1dTrend_VolumeConfirmation
Hypothesis: On 12h timeframe, Donchian(20) breakouts with 1d EMA50 trend filter and volume confirmation (>1.5x 24-bar avg) capture sustained moves in both bull and bear markets. Uses higher timeframe structure to reduce noise and avoid whipsaws. Targets 12-30 trades/year to minimize fee drag while maintaining edge via trend and volume filters.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (24-period = 12 days on 12h) for volume confirmation
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(60, 50, 24)  # 1d lookback, EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Calculate Donchian channels using 12h data (20 periods = 10 days)
            # We need to look back 20 periods from current bar (excluding current)
            lookback_start = max(0, i - 20)
            lookback_end = i  # exclude current bar to avoid look-ahead
            if lookback_end - lookback_start >= 20:
                highest_high = np.max(high[lookback_start:lookback_end])
                lowest_low = np.min(low[lookback_start:lookback_end])
                
                # Long: price breaks above Donchian high with uptrend and volume confirmation
                long_signal = (high_val > highest_high) and (close_val > ema_50_val) and volume_confirmed
                # Short: price breaks below Donchian low with downtrend and volume confirmation
                short_signal = (low_val < lowest_low) and (close_val < ema_50_val) and volume_confirmed
                
                if long_signal:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_val
                elif short_signal:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_val
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Trend reversal: close crosses below EMA50
            if close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Time-based exit: hold max 10 bars (5 days) to avoid mean reversion in chop
            elif i - (entry_price > 0 and 10 or 0) >= 10:  # simplified: exit after 10 bars
                # Actually track bars in trade
                pass  # We'll implement proper bar counting below
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Trend reversal: close crosses above EMA50
            if close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            # 2. Time-based exit: hold max 10 bars (5 days)
            elif i - (entry_price > 0 and 10 or 0) >= 10:
                pass
    
    # Fix: need to track bars in trade properly
    # Rewrite with proper tracking
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_in_trade = 0
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            bars_in_trade += 1
            continue
        
        # Get aligned values
        ema_50_val = ema_50_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume confirmation: current volume > 1.5x 24-period average
        volume_confirmed = vol_val > 1.5 * vol_ma_val
        
        # Calculate Donchian channels using 12h data (20 periods lookback)
        lookback_start = max(start_idx, i - 20)
        lookback_end = i  # exclude current bar
        if lookback_end - lookback_start >= 20:
            highest_high = np.max(high[lookback_start:lookback_end])
            lowest_low = np.min(low[lookback_start:lookback_end])
        else:
            highest_high = np.nan
            lowest_low = np.nan
        
        if position == 0:
            bars_in_trade = 0
            # Long: price breaks above Donchian high with uptrend and volume confirmation
            long_signal = (not np.isnan(highest_high)) and (high_val > highest_high) and (close_val > ema_50_val) and volume_confirmed
            # Short: price breaks below Donchian low with downtrend and volume confirmation
            short_signal = (not np.isnan(lowest_low)) and (low_val < lowest_low) and (close_val < ema_50_val) and volume_confirmed
            
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
            bars_in_trade += 1
            # Exit conditions:
            # 1. Trend reversal: close crosses below EMA50
            if close_val < ema_50_val:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            # 2. Time-based exit: hold max 12 bars (6 days) to avoid mean reversion in chop
            elif bars_in_trade >= 12:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_in_trade += 1
            # Exit conditions:
            # 1. Trend reversal: close crosses above EMA50
            if close_val > ema_50_val:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
            # 2. Time-based exit: hold max 12 bars (6 days)
            elif bars_in_trade >= 12:
                signals[i] = 0.0
                position = 0
                bars_in_trade = 0
    
    return signals

name = "12h_Donchian20_Breakout_1dTrend_VolumeConfirmation"
timeframe = "12h"
leverage = 1.0