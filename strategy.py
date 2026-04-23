#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend and volume confirmation
- Donchian(20) breakouts capture strong momentum moves in both bull and bear markets
- 1d EMA(34) ensures alignment with higher timeframe trend for multi-timeframe confirmation
- Volume > 1.8x 20-period average confirms breakout strength and reduces false signals
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with trend, in bear markets via fade of overextended moves
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 34, 20)  # Donchian, EMA1d, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels from lookback period (completed candles only)
        lookback_start = max(0, i - 20)
        lookback_end = i  # exclude current bar
        if lookback_end - lookback_start < 20:
            # Not enough data for Donchian calculation
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        highest_high = np.max(high[lookback_start:lookback_end])
        lowest_low = np.min(low[lookback_start:lookback_end])
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above Donchian upper + uptrend + volume spike
        # Short: price breaks below Donchian lower + downtrend + volume spike
        long_signal = (close[i] > highest_high and 
                      close[i] > ema_34_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < lowest_low and 
                       close[i] < ema_34_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend reversal or opposite Donchian level break
            exit_signal = False
            
            if position == 1:
                # Exit long: trend reversal or price breaks below Donchian lower
                if (close[i] < ema_34_1d_aligned[i] or 
                    close[i] < lowest_low):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Donchian upper
                if (close[i] > ema_34_1d_aligned[i] or 
                    close[i] > highest_high):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0