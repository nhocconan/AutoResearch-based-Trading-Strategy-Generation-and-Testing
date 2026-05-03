#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation.
# Long when: price breaks above 1d Donchian upper band (20-period high) AND close > 1w EMA50 AND volume > 2.0x 20-bar average
# Short when: price breaks below 1d Donchian lower band (20-period low) AND close < 1w EMA50 AND volume > 2.0x 20-bar average
# Exit via ATR(20) trailing stop: long exit when price < highest_high_since_entry - 2.5 * ATR
#                      short exit when price > lowest_low_since_entry + 2.5 * ATR
# Uses 1d Donchian for structure (proven edge from top performers), 1w EMA50 for HTF trend alignment, volume spike for confirmation
# Discrete sizing 0.25 balances return and fee drag. Target: 30-100 total trades over 4 years = 7-25/year.

name = "1d_Donchian20_1wEMA50_VolumeSpike_ATRStop_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA50 (HTF trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (structure)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR for trailing stop
    tr1 = pd.Series(high - low).values
    tr2 = pd.Series(np.abs(high - np.roll(close, 1))).values
    tr3 = pd.Series(np.abs(low - np.roll(close, 1))).values
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Track position state for trailing stop
    position_side = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0
    lowest_since_entry = 0
    
    for i in range(lookback, n):
        # Skip if EMA not available yet
        if np.isnan(ema_50_1w_aligned[i]):
            continue
            
        # Check for breakout conditions
        bull_breakout = close[i] > highest_high[i]
        bear_breakout = close[i] < lowest_low[i]
        volume_spike = volume[i] > 2.0 * avg_volume[i]
        
        # Check trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Exit conditions (trailing stop)
        if position_side == 1:  # long position
            highest_since_entry = max(highest_since_entry, high[i])
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0  # exit long
                position_side = 0
                continue
        elif position_side == -1:  # short position
            lowest_since_entry = min(lowest_since_entry, low[i])
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0  # exit short
                position_side = 0
                continue
        
        # Entry conditions
        if position_side == 0:  # only enter when flat
            if bull_breakout and uptrend and volume_spike:
                signals[i] = 0.25  # long 25%
                position_side = 1
                highest_since_entry = high[i]
            elif bear_breakout and downtrend and volume_spike:
                signals[i] = -0.25  # short 25%
                position_side = -1
                lowest_since_entry = low[i]
        else:
            # Hold current position
            signals[i] = 0.25 if position_side == 1 else -0.25
    
    return signals