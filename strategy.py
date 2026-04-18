#!/usr/bin/env python3
"""
4h_RSI_MeanReversion_v1
Strategy: 4h RSI mean-reversion with volume confirmation and trend filter.
Long: RSI < 30 + price > 200EMA + volume > 1.5x average.
Short: RSI > 70 + price < 200EMA + volume > 1.5x average.
Uses 1d trend filter to avoid counter-trend trades in strong trends.
Designed for 4h timeframe: ~20-30 trades/year per symbol (80-120 total over 4 years).
Works in bull/bear via trend filter and mean-reversion logic.
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    close_1d = df_1d['close'].values
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align daily EMA200 to 4h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # need enough for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema_200_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of daily trend
        uptrend = close[i] > ema_200_aligned[i]
        downtrend = close[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # RSI conditions
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: oversold + uptrend + volume
            if rsi_oversold and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume
            elif rsi_overbought and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI neutral or trend change
            if rsi[i] >= 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI neutral or trend change
            if rsi[i] <= 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_RSI_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0