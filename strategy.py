#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme Reversal with 1d EMA34 Trend Filter and Volume Spike
- Williams %R(14) identifies overbought/oversold conditions for mean reversion entries
- 1d EMA(34) ensures alignment with daily trend - only trade pullbacks in trend direction
- Volume > 2.0x 20-period average confirms reversal conviction
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in bull markets by buying pullbacks, in bear markets by selling rallies
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams %R(14) on 4h timeframe
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 14, 20)  # EMA1d, Williams %R, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R extreme reversal signals with trend filter and volume confirmation
        # Long: Williams %R < -80 (oversold) + price > EMA34(1d) (uptrend) + volume spike
        # Short: Williams %R > -20 (overbought) + price < EMA34(1d) (downtrend) + volume spike
        long_signal = (williams_r[i] < -80 and 
                      close[i] > ema_34_1d_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r[i] > -20 and 
                       close[i] < ema_34_1d_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R returns to neutral zone or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 (leaving oversold) or trend reversal
                if (williams_r[i] > -50 or 
                    close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R < -50 (leaving overbought) or trend reversal
                if (williams_r[i] < -50 or 
                    close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0