#!/usr/bin/env python3
"""
1d_rsi_200ma_breakout_v1
Hypothesis: On daily timeframe, go long when RSI(14) crosses above 50 while price > EMA200, short when RSI crosses below 50 while price < EMA200, with volume > 1.5x 20-day average for confirmation. Uses weekly EMA200 trend filter to avoid counter-trend trades. Target: 15-30 trades/year to minimize fee dust.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_rsi_200ma_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily EMA200 for trend filter
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate weekly EMA200 for higher timeframe trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_200[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        # Trend filter: price above/below weekly EMA200
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 or trend changes to downtrend
            if rsi[i] < 50 or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 or trend changes to uptrend
            if rsi[i] > 50 or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI crosses above 50 in uptrend
                if rsi[i] > 50 and rsi[i-1] <= 50 and uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI crosses below 50 in downtrend
                elif rsi[i] < 50 and rsi[i-1] >= 50 and downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals