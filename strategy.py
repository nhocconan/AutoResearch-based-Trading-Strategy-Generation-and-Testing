#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with 1d EMA50 trend filter + 6h RSI(14) mean reversion + volume spike confirmation.
Long when price > 1d EMA50 (uptrend) AND 6h RSI < 30 (oversold) AND 6h volume > 1.5x 20-period average.
Short when price < 1d EMA50 (downtrend) AND 6h RSI > 70 (overbought) AND 6h volume > 1.5x 20-period average.
Exit when RSI returns to neutral zone (40-60) or opposite extreme is hit.
Uses discrete position sizing 0.25 to limit fee drag. Target: 50-150 total trades over 4 years.
Combines trend following with mean reversion entries to work in both bull and bear markets.
Volume spike confirms institutional interest, reducing false signals.
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
    
    # Get daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 6h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate 6h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for EMA50 and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: uptrend + oversold + volume spike
            if (close[i] > ema50_1d_aligned[i] and 
                rsi_values[i] < 30 and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: downtrend + overbought + volume spike
            elif (close[i] < ema50_1d_aligned[i] and 
                  rsi_values[i] > 70 and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (40-60) or becomes overbought
            if rsi_values[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (40-60) or becomes oversold
            if rsi_values[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1dEMA50_RSI14_VolumeSpike"
timeframe = "6h"
leverage = 1.0