#!/usr/bin/env python3
"""
6h_LongTermTrend_Pullback_With_VolumeConfirmation
Hypothesis: Uses 6-hour price action with long-term trend filter from 1-day EMA200 and pullback entries.
Looks for pullbacks to the 6-hour EMA21 during strong daily trends with volume confirmation.
Designed to work in both bull and bear markets by following the dominant daily trend.
Targets 12-37 trades per year to minimize fee decay while capturing meaningful trend continuation moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200 for long-term trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 6-hour EMA21 for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate volume confirmation (>1.5x 20-period MA)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_21[i])):
            signals[i] = 0.0
            continue
        
        # Long-term trend direction from 1-day EMA200
        trend_up = close[i] > ema_200_1d_aligned[i]
        trend_down = close[i] < ema_200_1d_aligned[i]
        
        # Price relative to 6-hour EMA21
        price_above_ema21 = close[i] > ema_21[i]
        price_below_ema21 = close[i] < ema_21[i]
        
        # Entry logic:
        # Long: Pullback to EMA21 in uptrend with volume confirmation
        long_entry = trend_up and price_below_ema21 and vol_confirm[i] and (close[i] > close[i-1])
        
        # Short: Pullback to EMA21 in downtrend with volume confirmation
        short_entry = trend_down and price_above_ema21 and vol_confirm[i] and (close[i] < close[i-1])
        
        # Exit logic: Opposite EMA21 cross or trend reversal
        long_exit = price_above_ema21 or not trend_up
        short_exit = price_below_ema21 or not trend_down
        
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

name = "6h_LongTermTrend_Pullback_With_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0