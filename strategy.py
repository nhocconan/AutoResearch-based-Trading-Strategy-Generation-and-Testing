#!/usr/bin/env python3
"""
6h Triple Filter: 1d MACD Trend + 12h RSI Momentum + Volume Spike
Long: MACD bullish (MACD>Signal) + RSI(14)>55 + volume > 2x 6m volume SMA(20)
Short: MACD bearish (MACD<Signal) + RSI(14)<45 + volume > 2x 6m volume SMA(20)
Exit: Opposite MACD cross or RSI crosses 50
Uses MACD for trend direction, RSI for momentum filter, volume for confirmation.
Designed to work in both bull and bear markets by requiring trend-momentum alignment.
Target: 50-150 total trades over 4 years (12-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for MACD trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d MACD(12,26,9)
    ema12 = pd.Series(close_1d).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line  # Positive = bullish, Negative = bearish
    
    # Align MACD histogram to 6h timeframe
    macd_hist_aligned = align_htf_to_ltf(prices, df_1d, macd_hist)
    
    # Get 12h data for RSI momentum filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h RSI(14)
    delta = pd.Series(close_12h).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    rsi_12h = 100 - (100 / (1 + rs))
    
    # Align RSI to 6h timeframe
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    
    # Calculate 6m volume SMA(20) for volume filter
    vol_sma_6m = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need sufficient history for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(macd_hist_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or
            np.isnan(vol_sma_6m[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_6m[i]
        macd_h = macd_hist_aligned[i]
        rsi_val = rsi_12h_aligned[i]
        
        if position == 0:
            # Long: MACD bullish + RSI bullish momentum + volume spike
            if macd_h > 0 and rsi_val > 55 and vol > 2.0 * vol_sma_val:
                signals[i] = 0.25
                position = 1
            # Short: MACD bearish + RSI bearish momentum + volume spike
            elif macd_h < 0 and rsi_val < 45 and vol > 2.0 * vol_sma_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: MACD turns bearish or RSI loses momentum
            if macd_h < 0 or rsi_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: MACD turns bullish or RSI gains momentum
            if macd_h > 0 or rsi_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_TripleFilter_MACD_RSI_Volume"
timeframe = "6h"
leverage = 1.0