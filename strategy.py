#!/usr/bin/env python3
"""
4h_MultiTimeframe_Confluence_Bands
Hypothesis: Combines 4h Bollinger Band breakout with 1d EMA trend filter and volume confirmation.
Goes long when price breaks above upper Bollinger Band in uptrend with volume spike,
short when breaks below lower band in downtrend with volume spike.
Designed for low trade frequency (15-30 trades/year) to minimize fee drift while capturing
strong directional moves in both bull and bear markets by following daily trend.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=bb_period, min_periods=bb_period).mean().values
    bb_std_dev = close_series.rolling(window=bb_period, min_periods=bb_period).std().values
    bb_upper = bb_middle + (bb_std_dev * bb_std)
    bb_lower = bb_middle - (bb_std_dev * bb_std)
    
    # Volume confirmation: >1.8x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below daily EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Bollinger Band breakout conditions
        bb_breakout_up = close[i] > bb_upper[i]
        bb_breakout_down = close[i] < bb_lower[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Entry logic: breakout in direction of trend with volume
        long_entry = vol_confirm and uptrend and bb_breakout_up
        short_entry = vol_confirm and downtrend and bb_breakout_down
        
        # Exit logic: opposite breakout or trend change
        long_exit = bb_breakout_down or (not uptrend)
        short_exit = bb_breakout_up or (not downtrend)
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
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

name = "4h_MultiTimeframe_Confluence_Bands"
timeframe = "4h"
leverage = 1.0