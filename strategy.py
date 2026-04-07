#!/usr/bin/env python3
"""
4h_momentum_volume_trap_v1
Hypothesis: Combines momentum (MACD crossover) with volume spike (>2x average) and 12h trend filter to capture strong moves in both bull and bear markets. Uses tight RSI filters to avoid whipsaws and discrete position sizing (0.25) to minimize churn. Designed for low trade frequency (<30/year) with high win rate by requiring multiple confirmations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_momentum_volume_trap_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # MACD (12,26,9)
    fast = pd.Series(close).ewm(span=12, adjust=False).mean()
    slow = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd = fast - slow
    signal_line = pd.Series(macd).ewm(span=9, adjust=False).mean()
    macd_hist = macd - signal_line
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # RSI(14) for entry filter
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after MACD warmup
        # Skip if data not available
        if (np.isnan(macd_hist[i]) or np.isnan(signal_line[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 2x average volume (strict to reduce trades)
        vol_ok = volume[i] > (vol_ma[i] * 2.0)
        
        if position == 1:  # Long position
            # Exit: MACD histogram turns negative or trend changes
            if macd_hist[i] < 0 or close[i] < ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: MACD histogram turns positive or trend changes
            if macd_hist[i] > 0 or close[i] > ema_50_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: MACD bullish crossover + RSI > 50 (momentum) + uptrend
                if (macd_hist[i] > 0 and macd_hist[i-1] <= 0 and  # bullish cross
                    rsi[i] > 50 and 
                    close[i] > ema_50_12h_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: MACD bearish crossover + RSI < 50 (momentum) + downtrend
                elif (macd_hist[i] < 0 and macd_hist[i-1] >= 0 and  # bearish cross
                      rsi[i] < 50 and 
                      close[i] < ema_50_12h_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals