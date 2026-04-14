#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R with 1-day momentum filter and volume confirmation
# Williams %R identifies overbought/oversold conditions in ranging markets
# 1-day momentum (price change) filters for directional bias
# Volume > average confirms participation
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies)
# Target: 20-40 trades/year per symbol with disciplined entries

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for momentum and Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Williams %R (14 periods) on daily
    lookback = 14
    highest_high = pd.Series(df_1d['high']).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=lookback, min_periods=lookback).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1-day momentum (close to close change)
    daily_momentum = df_1d['close'].pct_change().values
    daily_momentum_aligned = align_htf_to_ltf(prices, df_1d, daily_momentum)
    
    # Calculate volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(daily_momentum_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R levels: oversold < -80, overbought > -20
        oversold = williams_r_aligned[i] < -80
        overbought = williams_r_aligned[i] > -20
        
        # Momentum filter: positive momentum for longs, negative for shorts
        mom_pos = daily_momentum_aligned[i] > 0
        mom_neg = daily_momentum_aligned[i] < 0
        
        # Volume confirmation: current volume > average
        volume_confirmed = volume[i] > vol_ma[i]
        
        if position == 0:
            # Enter long: Williams %R oversold + positive momentum + volume
            if oversold and mom_pos and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: Williams %R overbought + negative momentum + volume
            elif overbought and mom_neg and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Williams %R returns to neutral (> -50) or momentum turns negative
            if williams_r_aligned[i] > -50 or daily_momentum_aligned[i] < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Williams %R returns to neutral (< -50) or momentum turns positive
            if williams_r_aligned[i] < -50 or daily_momentum_aligned[i] > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_WilliamsR_Momentum_Volume_v1"
timeframe = "4h"
leverage = 1.0