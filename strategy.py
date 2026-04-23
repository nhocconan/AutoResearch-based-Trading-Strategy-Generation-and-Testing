#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversal with 1d EMA50 Trend Filter and Volume Confirmation
- Williams %R(14) identifies overbought/oversold conditions: < -80 = oversold, > -20 = overbought
- 1d EMA(50) provides higher timeframe trend filter for multi-timeframe alignment
- Volume > 1.8x 20-period average confirms momentum and reduces false reversal signals
- Designed for 4h timeframe targeting 30-60 trades/year (120-240 over 4 years) to balance opportunity and fee drag
- Works in bull markets via buying oversold dips in uptrend, in bear markets via selling overbought rallies in downtrend
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
    
    # Get 1d data for Williams %R calculation and EMA trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Williams %R on 1d timeframe: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    
    # Align Williams %R to 4h timeframe (completed 1d bar only)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA1d, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme reversal signals with trend filter and volume confirmation
        # Long: %R < -80 (oversold) + price above EMA (uptrend) + volume spike
        # Short: %R > -20 (overbought) + price below EMA (downtrend) + volume spike
        long_signal = (williams_r_aligned[i] < -80 and 
                      close[i] > ema_50_1d_aligned[i] and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (williams_r_aligned[i] > -20 and 
                       close[i] < ema_50_1d_aligned[i] and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone or opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: %R rises above -50 (leaving oversold) or becomes overbought
                if williams_r_aligned[i] > -50:
                    exit_signal = True
            elif position == -1:
                # Exit short: %R falls below -50 (leaving overbought) or becomes oversold
                if williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0