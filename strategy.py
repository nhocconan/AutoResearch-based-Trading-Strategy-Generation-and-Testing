#!/usr/bin/env python3
"""
6h_RSI_50_Cross_1wTrend_Signal
Hypothesis: Uses weekly trend (price above/below 200-week EMA) to determine directional bias, and enters on RSI(14) crossing above/below 50 on 6h chart with volume confirmation (>1.5x average). This captures momentum shifts in both bull and bear markets by aligning with higher timeframe trend. Targets 10-20 trades/year via strict weekly trend filter and RSI cross conditions.
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
    
    # Get weekly data for trend filter (200 EMA)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly EMA200 for trend filter
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # RSI(14) calculation
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume confirmation: >1.5x 50-period MA (~12.5 days of 6h bars)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for RSI and volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(rsi[i]) or
            np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter: price above/below 200-week EMA
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_50[i])
        
        # RSI cross conditions
        rsi_cross_up = (rsi[i] > 50) and (rsi[i-1] <= 50)
        rsi_cross_down = (rsi[i] < 50) and (rsi[i-1] >= 50)
        
        # Entry conditions
        long_entry = rsi_cross_up and vol_confirm and uptrend
        short_entry = rsi_cross_down and vol_confirm and downtrend
        
        # Exit conditions: opposite RSI cross
        long_exit = rsi_cross_down
        short_exit = rsi_cross_up
        
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

name = "6h_RSI_50_Cross_1wTrend_Signal"
timeframe = "6h"
leverage = 1.0