#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Williams %R extreme + 1d trend filter + volume spike
    # Enter long when Williams %R(14) < -80 (oversold) AND 1d EMA50 > EMA200 (uptrend) AND volume > 1.5x 20-bar avg
    # Enter short when Williams %R(14) > -20 (overbought) AND 1d EMA50 < EMA200 (downtrend) AND volume > 1.5x 20-bar avg
    # Exit when Williams %R returns to -50 (mean reversion) or opposite extreme
    # Williams %R identifies exhaustion points; 1d EMA filter ensures we trade with higher timeframe trend
    # Volume spike confirms participation at extremes
    # Target: 60-120 total trades over 4 years (15-30/year) to balance opportunity and fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for primary timeframe
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:  # need enough for EMA200
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R on 6h: (highest high - close) / (highest high - lowest low) * -100
    # Williams %R = -100 * (HHV - CLOSE) / (HHV - LLV)
    period = 14
    highest_high = pd.Series(high_6h).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low_6h).rolling(window=period, min_periods=period).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when highest_high == lowest_low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMAs to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(1, n):  # start from 1 to access previous bar
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        mean_reversion_exit = abs(williams_r[i] + 50) < 10  # near -50
        
        # 1d trend filter
        uptrend_1d = ema_50_1d_aligned[i] > ema_200_1d_aligned[i]
        downtrend_1d = ema_50_1d_aligned[i] < ema_200_1d_aligned[i]
        
        # Entry conditions
        long_entry = oversold and uptrend_1d and volume_confirmed[i] and position != 1
        short_entry = overbought and downtrend_1d and volume_confirmed[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (mean_reversion_exit or overbought))
        exit_short = (position == -1 and (mean_reversion_exit or oversold))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_williamsr_trend_volume_v1"
timeframe = "6h"
leverage = 1.0