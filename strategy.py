#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversal with 1d Trend Filter and Volume Spike.
- Williams %R identifies overbought/oversold conditions on 4h.
- Extreme readings (%R < -90 for long, %R > -10 for short) signal high-probability reversals.
- 1d EMA50 provides higher-timeframe trend filter to align with dominant trend.
- Volume spike (>2x 20-period average) confirms conviction behind the reversal.
- Position size 0.25 balances profit potential and drawdown control.
- Target trades: 100-200 total over 4 years (25-50/year) to balance opportunity and fee drag.
- Works in bull/bear markets via 1d trend filter and mean-reversion logic.
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
    
    # Get 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams %R (14-period) on 4h
    wr_period = 14
    highest_high = pd.Series(high).rolling(window=wr_period, min_periods=wr_period).max().values
    lowest_low = pd.Series(low).rolling(window=wr_period, min_periods=wr_period).min().values
    wr = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    wr = np.where((highest_high - lowest_low) == 0, -50, wr)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(wr_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(wr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme conditions
        wr_oversold = wr[i] < -90  # Extremely oversold
        wr_overbought = wr[i] > -10  # Extremely overbought
        
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Only enter on extreme readings with volume confirmation
            if volume_confirm:
                # Long: extremely oversold + above 1d EMA50 (bullish higher-timeframe trend)
                if wr_oversold and close[i] > ema_50_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short: extremely overbought + below 1d EMA50 (bearish higher-timeframe trend)
                elif wr_overbought and close[i] < ema_50_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: Williams %R returns above -50 (momentum fading) OR reverse signal
            if wr[i] > -50 or wr_overbought:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Williams %R returns below -50 (momentum fading) OR reverse signal
            if wr[i] < -50 or wr_oversold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0