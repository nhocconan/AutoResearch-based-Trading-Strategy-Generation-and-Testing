#!/usr/bin/env python3
"""
4h Camarilla R3S3 Breakout + 1d EMA34 Trend + Volume Spike
Hypothesis: Camarilla R3/S3 levels act as strong support/resistance derived from prior day's range.
Breakouts above R3 (bullish) or below S3 (bearish) with volume confirmation indicate institutional
participation. Trend filter (price vs 1d EMA34) ensures we trade with the higher timeframe momentum.
This strategy targets 20-50 trades/year on 4h timeframe (80-200 over 4 years), balancing opportunity
with cost efficiency. Works in bull markets (buy breakouts in uptrend) and bear markets (sell breakdowns
in downtrend) by only taking trades aligned with the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar
    # R3 = close + 1.1*(high - low)/2
    # S3 = close - 1.1*(high - low)/2
    camarilla_range = df_1d['high'] - df_1d['low']
    r3 = df_1d['close'] + (camarilla_range * 1.1 / 2)
    s3 = df_1d['close'] - (camarilla_range * 1.1 / 2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for calculations
    start_idx = max(20, 34)  # volume MA, 1d EMA34 alignment
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price relative to 1d EMA34
        uptrend = curr_close > ema_34_aligned[i]
        downtrend = curr_close < ema_34_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: Close > R3 AND uptrend AND volume spike
            long_entry = (curr_close > curr_r3) and uptrend and vol_spike
            # Short: Close < S3 AND downtrend AND volume spike
            short_entry = (curr_close < curr_s3) and downtrend and vol_spike
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: Close drops below R3 (breakout failure) OR loss of uptrend
            if (curr_close < curr_r3) or (curr_close < ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: Close rises above S3 (breakdown failure) OR loss of downtrend
            if (curr_close > curr_s3) or (curr_close > ema_34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0