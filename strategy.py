#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Donchian(20) breakout captures medium-term momentum with clear structure
- 1w EMA50 ensures alignment with strong weekly trend for multi-timeframe confirmation
- Volume > 1.5x 20-period average confirms breakout momentum with moderate filtering
- Designed for 1d timeframe targeting 15-30 trades/year (60-120 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with strong trend, in bear markets via fade of overextended moves at strong levels
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
    
    # Get 1d data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian(20) channels from 1d data (using prior 20 completed days)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (completed 1d bar only)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 1w EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 1d timeframe (completed 1w bar only)
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-period average (moderate filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA needs 50, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above Donchian high + price above weekly EMA50 (uptrend) + volume confirmation
        # Short: price breaks below Donchian low + price below weekly EMA50 (downtrend) + volume confirmation
        long_signal = (close[i] > donchian_high_aligned[i] and 
                      close[i] > ema_50_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low_aligned[i] and 
                       close[i] < ema_50_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal (price crosses weekly EMA50) or opposite Donchian break
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below weekly EMA50 or breaks below Donchian low
                if (close[i] < ema_50_aligned[i] or 
                    close[i] < donchian_low_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above weekly EMA50 or breaks above Donchian high
                if (close[i] > ema_50_aligned[i] or 
                    close[i] > donchian_high_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0