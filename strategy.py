#!/usr/bin/env python3
"""
102443
4h_Chaikin_Money_Flow_Donchian_20_Breakout_12hTrend
Hypothesis: Combines Donchian(20) breakout with 12h trend filter (EMA50) and Chaikin Money Flow (CMF) volume confirmation. CMF > 0 confirms accumulation on breakouts. Designed for 15-25 trades/year via strict confluence of price breakout, trend alignment, and volume confirmation. Works in bull/bear by following 12h trend direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate Chaikin Money Flow (20-period)
    # Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    mfm = ((close - low) - (high - close)) / (high - low)
    # Replace division by zero or NaN with 0
    mfm = np.where((high - low) == 0, 0, mfm)
    # Money Flow Volume = MFM * Volume
    mfv = mfm * volume
    # CMF = 20-period sum of MFV / 20-period sum of volume
    mfv_sum = pd.Series(mfv).rolling(window=20, min_periods=20).sum().values
    vol_sum = pd.Series(volume).rolling(window=20, min_periods=20).sum().values
    cmf = np.where(vol_sum != 0, mfv_sum / vol_sum, 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for EMA50 and Donchian to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(cmf[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation: CMF > 0 (accumulation)
        vol_confirm = cmf[i] > 0
        
        # Breakout conditions: price breaks Donchian channels
        long_breakout = close[i] > highest_high[i] and vol_confirm and uptrend
        short_breakout = close[i] < lowest_low[i] and vol_confirm and downtrend
        
        # Exit conditions: return to opposite Donchian level
        long_exit = close[i] < lowest_low[i]
        short_exit = close[i] > highest_high[i]
        
        if long_breakout and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Chaikin_Money_Flow_Donchian_20_Breakout_12hTrend"
timeframe = "4h"
leverage = 1.0