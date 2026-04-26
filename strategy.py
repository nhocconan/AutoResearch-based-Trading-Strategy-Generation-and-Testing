#!/usr/bin/env python3
"""
6h_RSI_Trend_Confluence
Hypothesis: On 6h timeframe, RSI(14) combined with 1d EMA trend filter and volume confirmation creates high-probability entries.
Long: RSI crosses above 40 in 1d uptrend with volume spike. Short: RSI crosses below 60 in 1d downtrend with volume spike.
Uses 1d EMA34 for trend alignment to work in both bull/bear markets. Volume spike ensures conviction.
Target: 12-37 trades/year per symbol (50-150 total over 4 years) to minimize fee drag.
Timeframe: 6h, HTF: 1d
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate RSI(14) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume spike detector (20-bar volume MA on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from daily EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # RSI edge detection
        rsi_above_40 = rsi_values[i] > 40 and rsi_values[i-1] <= 40
        rsi_below_60 = rsi_values[i] < 60 and rsi_values[i-1] >= 60
        
        if position == 0:
            # Long: RSI crosses above 40 in 1d uptrend with volume spike
            if rsi_above_40 and uptrend and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: RSI crosses below 60 in 1d downtrend with volume spike
            elif rsi_below_60 and downtrend and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: RSI crosses below 40 OR trend change to downtrend
            if rsi_values[i] < 40 or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: RSI crosses above 60 OR trend change to uptrend
            if rsi_values[i] > 60 or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_RSI_Trend_Confluence"
timeframe = "6h"
leverage = 1.0