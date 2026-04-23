#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R Extreme + 1d EMA50 Trend + Volume Spike
- Long: Williams %R(14) < -80 (oversold) + price > 1d EMA50 (uptrend) + volume > 2.0x 20-period average
- Short: Williams %R(14) > -20 (overbought) + price < 1d EMA50 (downtrend) + volume > 2.0x 20-period average
- Exit: Williams %R crosses back above -50 (long) or below -50 (short) OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 25-50 trades/year (100-200 over 4 years) to avoid fee drag
- Williams %R identifies exhaustion points; works in bull (buy oversold dips in uptrend) and bear (sell overbought rallies in downtrend)
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Williams %R(14) on 4h data
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 14, 20)  # EMA50 needs 50, Williams %R needs 14, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        close_1d = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema50_aligned[i]
        downtrend = close_1d_aligned[i] < ema50_aligned[i]
        
        # Williams %R signals with trend filter and volume confirmation
        # Long: Williams %R < -80 (oversold) + uptrend + volume spike
        # Short: Williams %R > -20 (overbought) + downtrend + volume spike
        long_signal = (williams_r[i] < -80 and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (williams_r[i] > -20 and 
                       downtrend and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses back above -50 (long) or below -50 (short) OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R > -50 or trend turns down
                if (williams_r[i] > -50 or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Williams %R < -50 or trend turns up
                if (williams_r[i] < -50 or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0