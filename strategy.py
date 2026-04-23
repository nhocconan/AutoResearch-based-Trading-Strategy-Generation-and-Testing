#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
- Long: Close breaks above Camarilla R3 + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Close breaks below Camarilla S3 + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Close retouches Camarilla Pivot (PP) OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 25-50 trades/year (100-200 over 4 years) to avoid fee drag
- Camarilla levels provide institutional support/resistance; breakouts with volume work in both bull and bear markets
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
    
    # Get 1d data for EMA34 trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Camarilla: PP = (H+L+C)/3, R3 = C + (H-L)*1.1/2, S3 = C - (H-L)*1.1/2
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels
    pp = (prev_high + prev_low + prev_close) / 3.0
    r3 = prev_close + (prev_high - prev_low) * 1.1 / 2.0
    s3 = prev_close - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA34 needs 34+1 for shift, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(pp_aligned[i]) or 
            np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Close breaks above R3 + uptrend + volume spike
        # Short: Close breaks below S3 + downtrend + volume spike
        long_signal = (close[i] > r3_aligned[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < s3_aligned[i] and 
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
            # Exit conditions: Close retouches Pivot Point (PP) OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Close retouches PP or trend turns down
                if (close[i] <= pp_aligned[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Close retouches PP or trend turns up
                if (close[i] >= pp_aligned[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0