#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w Supertrend(ATR=10,mult=3) + volume confirmation
# In trending markets (Supertrend long), we trade breakouts in trend direction: long on upper Donchian breakout, short on lower Donchian breakout.
# In ranging markets (Supertrend flat), we fade Donchian extremes: short near upper band, long near lower band.
# Volume confirmation (>1.3x 20-period EMA) reduces false breakouts. Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "12h_Donchian20_1wSupertrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Supertrend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1w ATR(10)
    tr1 = pd.Series(df_1w['high']).sub(df_1w['low'])
    tr2 = pd.Series(df_1w['high']).sub(df_1w['close'].shift(1)).abs()
    tr3 = pd.Series(df_1w['low']).sub(df_1w['close'].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=10, min_periods=10).mean()
    
    # Calculate 1w Supertrend
    hl2 = (df_1w['high'] + df_1w['low']) / 2
    upperband = hl2 + (3 * atr)
    lowerband = hl2 - (3 * atr)
    
    supertrend = np.zeros(len(df_1w))
    direction = np.ones(len(df_1w))  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(df_1w)):
        if close_1w := df_1w['close'].iloc[i]:
            pass
        # Current close
        close_curr = df_1w['close'].iloc[i]
        
        # Upper/lower band calculation
        upperband_curr = hl2.iloc[i] + (3 * atr.iloc[i])
        lowerband_curr = hl2.iloc[i] - (3 * atr.iloc[i])
        
        # Supertrend logic
        if i == 1:
            supertrend[i] = lowerband_curr
            direction[i] = 1
        else:
            if supertrend[i-1] == upperband.iloc[i-1]:
                supertrend[i] = lowerband_curr if close_curr > upperband_curr else upperband_curr
                direction[i] = -1 if supertrend[i] == upperband_curr else 1
            else:
                supertrend[i] = upperband_curr if close_curr < lowerband_curr else lowerband_curr
                direction[i] = 1 if supertrend[i] == lowerband_curr else -1
    
    # Simplified Supertrend: just use direction
    supertrend_direction = direction
    
    # Calculate 12h Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Align 1w Supertrend direction to 12h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, supertrend_direction.astype(float))
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(lookback, n):
        # Skip if any value is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine market state: trending (Supertrend active) or ranging
            if supertrend_dir_aligned[i] != 0:
                # Trending market: trade breakouts in trend direction
                if supertrend_dir_aligned[i] == 1:  # Uptrend
                    if close[i] > highest_high[i] and volume_confirm:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend
                    if close[i] < lowest_low[i] and volume_confirm:
                        signals[i] = -0.25
                        position = -1
            else:
                # Ranging market: fade Donchian extremes
                if close[i] <= lowest_low[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= highest_high[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches lower Donchian band OR Supertrend turns bearish
            if close[i] <= lowest_low[i] or supertrend_dir_aligned[i] == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches upper Donchian band OR Supertrend turns bullish
            if close[i] >= highest_high[i] or supertrend_dir_aligned[i] == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals