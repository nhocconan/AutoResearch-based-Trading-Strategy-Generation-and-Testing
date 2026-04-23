#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Daily Donchian breakouts capture medium-term trends with fewer trades than lower timeframes
- Weekly EMA(50) ensures alignment with longer-term trend to avoid counter-trend entries
- Volume > 1.8x 20-period average confirms breakout strength
- Designed for 1d timeframe targeting 7-25 trades/year (30-100 over 4 years) to minimize fee drag
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Donchian(20) on daily data: upper = 20-period high, lower = 20-period low
    # Use previous day's data to avoid look-ahead (breakout of prior 20-day range)
    donchian_upper = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_lower = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1d timeframe (completed 1d bar only)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    
    # Get 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA1w, Donchian20, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: price breaks above Donchian upper + uptrend + volume confirmation
        # Short: price breaks below Donchian lower + downtrend + volume confirmation
        long_signal = (close[i] > donchian_upper_aligned[i] and 
                      close[i] > ema_50_1w_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < donchian_lower_aligned[i] and 
                       close[i] < ema_50_1w_aligned[i] and
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
                if (close[i] < ema_50_1w_aligned[i] or 
                    close[i] < donchian_lower_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: trend reversal or price breaks above Donchian upper
                if (close[i] > ema_50_1w_aligned[i] or 
                    close[i] > donchian_upper_aligned[i]):
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