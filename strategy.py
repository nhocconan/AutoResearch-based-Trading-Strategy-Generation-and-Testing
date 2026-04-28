#!/usr/bin/env python3
"""
6h_RSI_Extreme_12hTrend_VolumeSpike
Hypothesis: Extreme RSI (14) values on 6h (RSI<20 or >80) with 12h trend filter (price>SMA50 for long, <SMA50 for short) and volume spike confirmation. Works in both bull/bear by fading extremes in trend direction. Targets 15-25 trades/year.
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h SMA50 for trend
    sma_50_12h = pd.Series(df_12h['close'].values).rolling(window=50, min_periods=50).mean().values
    sma_50_12h_aligned = align_htf_to_ltf(prices, df_12h, sma_50_12h)
    
    # Calculate 6h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: >2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Wait for RSI and SMA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(rsi[i]) or np.isnan(sma_50_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # 12h trend filter
        trend_up = close[i] > sma_50_12h_aligned[i]
        trend_down = close[i] < sma_50_12h_aligned[i]
        
        # RSI extremes
        rsi_oversold = rsi[i] < 20
        rsi_overbought = rsi[i] > 80
        
        # Volume confirmation
        vol_confirm = volume[i] > (2.0 * vol_ma_20[i])
        
        # Entry logic: fade RSI extremes in direction of 12h trend
        long_entry = vol_confirm and trend_up and rsi_oversold
        short_entry = vol_confirm and trend_down and rsi_overbought
        
        # Exit logic: RSI returns to neutral zone or trend reversal
        long_exit = (rsi[i] > 50) or (not trend_up)
        short_exit = (rsi[i] < 50) or (not trend_down)
        
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

name = "6h_RSI_Extreme_12hTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0