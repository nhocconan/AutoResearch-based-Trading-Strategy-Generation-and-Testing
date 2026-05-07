#!/usr/bin/env python3
"""
6H_RSI_Recovery_With_12hTrend_v1
Hypothesis: In 6B timeframe, buy when RSI(14) recovers from oversold (<30) in alignment with 12h uptrend (EMA50),
and sell/short when RSI becomes overbought (>70) in 12h downtrend. Uses volume confirmation to avoid false signals.
Designed to work in both bull (buy dips) and bear (sell rallies) markets by following higher timeframe trend.
"""
name = "6H_RSI_Recovery_With_12hTrend_v1"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(50) for trend
    close_12h = pd.Series(df_12h['close'])
    ema_12h = close_12h.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate RSI(14) on 6h close
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Volume filter: current volume > 1.3 * 20-period average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_exit = 0  # bars since last exit to prevent overtrading
    
    start_idx = max(30, 20)  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        bars_since_exit += 1
        
        # Skip if any data is not ready
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            continue
        
        if position == 0:
            # Minimum 8 bars between trades (~1.3 days on 6h TF) to reduce frequency
            if bars_since_exit < 8:
                continue
                
            # Long: RSI recovers from oversold (<30) in 12h uptrend
            if (rsi[i] > 30 and rsi[i-1] <= 30 and 
                close[i] > ema_12h_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
                bars_since_exit = 0
            # Short: RSI declines from overbought (>70) in 12h downtrend
            elif (rsi[i] < 70 and rsi[i-1] >= 70 and 
                  close[i] < ema_12h_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
                bars_since_exit = 0
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60) or trend reversal
            if position == 1 and (rsi[i] >= 60 or close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            elif position == -1 and (rsi[i] <= 40 or close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals