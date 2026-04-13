#!/usr/bin/env python3
"""
Hypothesis: 4h EMA trend filter with 1-day Bollinger Band mean reversion.
Uses 4h EMA(50) for trend direction and 1-day Bollinger Bands (20,2) for mean reversion entries.
Long when price touches lower BB in uptrend (EMA50 rising), short when touches upper BB in downtrend.
Adds volume confirmation (volume > 1.5x 20-period average) to avoid false signals.
Targets 50-150 total trades over 4 years (12-38/year) to balance opportunity and fee drag.
Works in bull/bear: trend filter avoids counter-trend trades in strong moves, BB mean reversion works in ranges.
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
    
    # 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_rising = np.diff(ema_50, prepend=ema_50[0]) > 0  # Rising if current > previous
    
    # 1-day Bollinger Bands (20,2)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    bb_middle = pd.Series(close_1d).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(close_1d).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    bb_upper_aligned = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_aligned = align_htf_to_ltf(prices, df_1d, bb_lower)
    
    # Volume confirmation (volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_rising[i]) or 
            np.isnan(bb_upper_aligned[i]) or 
            np.isnan(bb_lower_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions
        long_entry = (close[i] <= bb_lower_aligned[i]) and ema_rising[i] and vol_confirm[i]
        short_entry = (close[i] >= bb_upper_aligned[i]) and (not ema_rising[i]) and vol_confirm[i]
        
        # Exit when price crosses middle Bollinger Band
        exit_long = position == 1 and close[i] >= bb_middle[i - len(close) + len(df_1d) * 4]  # Approximate 4h bar index
        exit_short = position == -1 and close[i] <= bb_middle[i - len(close) + len(df_1d) * 4]
        
        # Simpler exit: when price touches opposite band or crosses EMA
        exit_long = position == 1 and (close[i] >= bb_upper_aligned[i] or not ema_rising[i])
        exit_short = position == -1 and (close[i] <= bb_lower_aligned[i] or ema_rising[i])
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
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

name = "4h_ema_bb_mean_reversion"
timeframe = "4h"
leverage = 1.0