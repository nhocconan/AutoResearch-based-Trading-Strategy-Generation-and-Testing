#!/usr/bin/env python3
"""
4h_adaptive_trend_pullback_v1
Hypothesis: Combines trend detection via 1d EMA50 with momentum confirmation via RSI(14) and volume surge (>1.5x average). Enters long on pullbacks to RSI 40-50 in uptrends and short on pullbacks to RSI 50-60 in downtrends. Uses discrete position sizing (0.25) to minimize churn. Designed to work in both bull (trend continuation) and bear (counter-trend bounces within larger trend) markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adaptive_trend_pullback_v1"
timeframe = "4h"
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
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):  # Start after RSI warmup
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: > 1.5x average volume
        vol_ok = volume[i] > (vol_ma[i] * 1.5)
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 40 (end of pullback) or trend changes
            if rsi[i] < 40 or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 60 (end of pullback) or trend changes
            if rsi[i] > 60 or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI pullback to 40-50 in uptrend (price above 1d EMA50)
                if (40 <= rsi[i] <= 50 and 
                    close[i] > ema_50_1d_aligned[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: RSI pullback to 50-60 in downtrend (price below 1d EMA50)
                elif (50 <= rsi[i] <= 60 and 
                      close[i] < ema_50_1d_aligned[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals