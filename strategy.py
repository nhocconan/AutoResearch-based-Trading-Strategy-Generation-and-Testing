#!/usr/bin/env python3
"""
1h_RSI2_Recovery_Volume_TrendFilter
1h strategy using 2-period RSI for mean-reversion entries with volume confirmation and 4h trend filter.
- Long: RSI(2) < 10 + volume > 1.5x 20-bar avg + price > 4h EMA50
- Short: RSI(2) > 90 + volume > 1.5x 20-bar avg + price < 4h EMA50
- Exit: RSI(2) > 60 for longs, RSI(2) < 40 for shorts (mean-reversion completion)
Designed for ~15-35 trades/year per symbol (60-140 total over 4 years)
Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 2-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[1] = gain[1]
    avg_loss[1] = loss[1]
    for i in range(2, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 1 + gain[i]) / 2  # 2-period
        avg_loss[i] = (avg_loss[i-1] * 1 + loss[i]) / 2
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # neutral for first value
    
    # Volume confirmation: 20-bar average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need volume MA and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions from 4h
        uptrend = close[i] > ema_50_4h_aligned[i]
        downtrend = close[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        # RSI conditions for mean reversion
        rsi_oversold = rsi[i] < 10
        rsi_overbought = rsi[i] > 90
        rsi_exit_long = rsi[i] > 60
        rsi_exit_short = rsi[i] < 40
        
        if position == 0:
            # Long: oversold RSI + volume + uptrend
            if rsi_oversold and vol_confirm and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: overbought RSI + volume + downtrend
            elif rsi_overbought and vol_confirm and downtrend:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: RSI recovery or trend change
            if rsi_exit_long or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: RSI recovery or trend change
            if rsi_exit_short or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_RSI2_Recovery_Volume_TrendFilter"
timeframe = "1h"
leverage = 1.0