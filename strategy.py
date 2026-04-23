#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout with 1w Supertrend Filter and Volume Spike
- Donchian(20) breakouts capture strong momentum moves in both bull and bear markets
- 1w Supertrend(ATR=10, mult=3.0) ensures alignment with major weekly trend to avoid counter-trend whipsaws
- Volume > 1.8x 20-period average confirms breakout momentum with moderate filtering
- Designed for 4h timeframe targeting 25-40 trades/year (100-160 over 4 years) to minimize fee drag
- Works in bull markets via breakouts with strong weekly uptrend, in bear markets via shorting breakdowns with strong weekly downtrend
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
    
    # Get 1w data for Supertrend trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Supertrend on weekly data
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['close']).shift(1)).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close']).shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean()
    
    # Basic Upper and Lower Bands
    basic_ub = (df_1w['high'] + df_1w['low']) / 2 + 3.0 * atr
    basic_lb = (df_1w['high'] + df_1w['low']) / 2 - 3.0 * atr
    
    # Final Upper and Lower Bands
    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()
    
    for i in range(1, len(df_1w)):
        if df_1w['close'].iloc[i-1] > final_ub.iloc[i-1]:
            final_ub.iloc[i] = basic_ub.iloc[i]
        else:
            final_ub.iloc[i] = min(basic_ub.iloc[i], final_ub.iloc[i-1])
            
        if df_1w['close'].iloc[i-1] < final_lb.iloc[i-1]:
            final_lb.iloc[i] = basic_lb.iloc[i]
        else:
            final_lb.iloc[i] = max(basic_lb.iloc[i], final_lb.iloc[i-1])
    
    # Supertrend direction
    supertrend = np.zeros(len(df_1w))
    supertrend[:] = np.nan
    for i in range(len(df_1w)):
        if i == 0:
            supertrend[i] = 1  # Start with uptrend assumption
        else:
            if supertrend[i-1] == 1:
                if df_1w['close'].iloc[i] <= final_lb.iloc[i]:
                    supertrend[i] = -1  # Reverse to downtrend
                else:
                    supertrend[i] = 1   # Continue uptrend
            else:  # supertrend[i-1] == -1
                if df_1w['close'].iloc[i] >= final_ub.iloc[i]:
                    supertrend[i] = 1   # Reverse to uptrend
                else:
                    supertrend[i] = -1  # Continue downtrend
    
    # Align Supertrend to 4h timeframe (completed 1w bar only)
    supertrend_aligned = align_htf_to_ltf(prices, df_1w, supertrend)
    
    # Donchian(20) channels on 4h data
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Supertrend needs ~30 weekly bars, Donchian 20, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with weekly Supertrend filter and volume spike
        # Long: price breaks above Donchian upper + weekly uptrend (Supertrend=1) + volume spike
        # Short: price breaks below Donchian lower + weekly downtrend (Supertrend=-1) + volume spike
        long_signal = (close[i] > high_ma[i] and 
                      supertrend_aligned[i] == 1 and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < low_ma[i] and 
                       supertrend_aligned[i] == -1 and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian break or weekly trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below Donchian lower or weekly trend turns down
                if (close[i] < low_ma[i] or 
                    supertrend_aligned[i] == -1):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian upper or weekly trend turns up
                if (close[i] > high_ma[i] or 
                    supertrend_aligned[i] == 1):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1wSupertrend_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0