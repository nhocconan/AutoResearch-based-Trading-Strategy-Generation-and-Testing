#!/usr/bin/env python3
"""
6h 200 EMA Trend + RSI Pullback + Volume Spike
Long: Price > 200 EMA, RSI < 40, volume > 2x avg volume
Short: Price < 200 EMA, RSI > 60, volume > 2x avg volume
Exit: Opposite signal or RSI crosses 50
Uses EMA for trend direction, RSI for pullback entry, volume for confirmation.
Designed to work in both bull and bear markets by trading with the trend.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 200 EMA for trend filter (on 6h data)
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate RSI(14) for pullback signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume SMA(20) for volume filter
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 200  # need EMA200 and RSI warmup
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_20[i]
        ema_val = ema_200[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: Price > EMA200 (uptrend), RSI < 40 (pullback), volume spike
            if price > ema_val and rsi_val < 40 and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: Price < EMA200 (downtrend), RSI > 60 (pullback), volume spike
            elif price < ema_val and rsi_val > 60 and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price < EMA200 (trend change) or RSI > 50 (pullback complete)
            if price < ema_val or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price > EMA200 (trend change) or RSI < 50 (pullback complete)
            if price > ema_val or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_EMA200_RSI_Pullback_Volume"
timeframe = "6h"
leverage = 1.0