#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot R3/S3 Breakout + 1d EMA34 Trend + Volume Spike
- Long: Price breaks above Camarilla R3 + price > 1d EMA34 (uptrend) + volume > 2.0x 20-period average
- Short: Price breaks below Camarilla S3 + price < 1d EMA34 (downtrend) + volume > 2.0x 20-period average
- Exit: Opposite Camarilla breakout (R4/S4) or trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 25-50 trades/year (100-200 over 4 years) to avoid fee drag
- Camarilla levels provide institutional support/resistance; works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend)
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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    camarilla_r3 = df_1d['close'] + 1.125 * (df_1d['high'] - df_1d['low'])
    camarilla_s3 = df_1d['close'] - 1.125 * (df_1d['high'] - df_1d['low'])
    camarilla_r4 = df_1d['close'] + 1.5 * (df_1d['high'] - df_1d['low'])
    camarilla_s4 = df_1d['close'] - 1.5 * (df_1d['high'] - df_1d['low'])
    
    # Align Camarilla levels to 4h timeframe (completed 1d bar only)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4.values)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4.values)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, volume MA 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        close_1d = df_1d['close'].values
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        uptrend = close_1d_aligned[i] > ema34_aligned[i]
        downtrend = close_1d_aligned[i] < ema34_aligned[i]
        
        # Camarilla breakout signals with trend filter and volume confirmation
        # Long: Price > Camarilla R3 + uptrend + volume spike
        # Short: Price < Camarilla S3 + downtrend + volume spike
        long_signal = (close[i] > camarilla_r3_aligned[i] and 
                      uptrend and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < camarilla_s3_aligned[i] and 
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
            # Exit conditions: Opposite Camarilla breakout (R4/S4) or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Price breaks above Camarilla R4 or trend turns down
                if (close[i] > camarilla_r4_aligned[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Price breaks below Camarilla S4 or trend turns up
                if (close[i] < camarilla_s4_aligned[i] or 
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