#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 1d EMA200 Trend Filter and Volume Spike
- Williams %R(14) identifies overextended moves: < -80 = oversold, > -20 = overbought
- Extreme readings (< -90 or > -10) often precede mean-reversion bounces in ranging/choppy markets
- 1d EMA200 filter ensures alignment with long-term trend: only long above EMA200, short below
- Volume > 1.5x 20-period average confirms reversal momentum
- Designed for 6h timeframe targeting 12-25 trades/year (50-100 over 4 years) to minimize fee drag
- Works in bull markets via pullbacks to EMA200, in bear markets via bounces from extreme %R levels
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA200 to 6h timeframe (completed 1d bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Calculate Williams %R(14) on 6h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(200, 20)  # EMA200 needs 200 bars, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_200_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme reversal signals with trend filter and volume spike
        # Long: Williams %R < -90 (extreme oversold) + price above 1d EMA200 + volume spike
        # Short: Williams %R > -10 (extreme overbought) + price below 1d EMA200 + volume spike
        long_signal = (williams_r[i] < -90 and 
                      close[i] > ema_200_aligned[i] and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (williams_r[i] > -10 and 
                       close[i] < ema_200_aligned[i] and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone (-50) or opposite extreme
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R returns above -50 or reaches extreme overbought
                if (williams_r[i] > -50 or 
                    williams_r[i] > -10):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R returns below -50 or reaches extreme oversold
                if (williams_r[i] < -50 or 
                    williams_r[i] < -90):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dEMA200_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0