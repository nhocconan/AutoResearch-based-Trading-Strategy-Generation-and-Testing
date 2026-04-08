#!/usr/bin/env python3
# 1h_4h1d_momentum_v1
# Hypothesis: Combines 4h EMA trend filter with 1d RSI momentum and 1h breakout entries.
# In bull markets: 4h EMA trend up + 1d RSI > 50 + 1h breakout above recent high.
# In bear markets: 4h EMA trend down + 1d RSI < 50 + 1h breakdown below recent low.
# Uses volume confirmation and session filter (08-20 UTC) to reduce noise.
# Target: 15-35 trades/year on 1h timeframe to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h EMA for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema4h_aligned = align_htf_to_ltf(prices, df_4h, ema4h)
    
    # 1d RSI for momentum filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi1d = 100 - (100 / (1 + rs))
    rsi1d_aligned = align_htf_to_ltf(prices, df_1d, rsi1d)
    
    # 1h ATR for volatility and breakout threshold
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 1h volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema4h_aligned[i]) or np.isnan(rsi1d_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                pass  # Hold position outside session
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        # 1h breakout levels (lookback 20 periods)
        lookback = 20
        if i >= lookback:
            high_lookback = np.max(high[i-lookback:i])
            low_lookback = np.min(low[i-lookback:i])
        else:
            high_lookback = high[i]
            low_lookback = low[i]
        
        if position == 1:  # Long position
            # Exit: 4h EMA trend down OR 1d RSI < 40 OR breakdown below recent low
            if (ema4h_aligned[i] < ema4h_aligned[i-1] or 
                rsi1d_aligned[i] < 40 or 
                close[i] < low_lookback):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: 4h EMA trend up OR 1d RSI > 60 OR breakout above recent high
            if (ema4h_aligned[i] > ema4h_aligned[i-1] or 
                rsi1d_aligned[i] > 60 or 
                close[i] > high_lookback):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long entry: 4h EMA trending up + 1d RSI > 50 + breakout above recent high + volume surge
            if (ema4h_aligned[i] > ema4h_aligned[i-1] and 
                rsi1d_aligned[i] > 50 and 
                close[i] > high_lookback and 
                vol_surge):
                position = 1
                signals[i] = 0.20
            # Short entry: 4h EMA trending down + 1d RSI < 50 + breakdown below recent low + volume surge
            elif (ema4h_aligned[i] < ema4h_aligned[i-1] and 
                  rsi1d_aligned[i] < 50 and 
                  close[i] < low_lookback and 
                  vol_surge):
                position = -1
                signals[i] = -0.20
    
    return signals