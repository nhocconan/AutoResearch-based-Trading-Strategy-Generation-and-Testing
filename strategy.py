# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h RSI(14) mean reversion with 4h Supertrend filter and volume confirmation
# Uses 4h Supertrend (ATR=10, mult=3) for trend bias and 1h RSI for entry timing
# Long when RSI<30 in uptrend (Supertrend up), short when RSI>70 in downtrend
# Volume filter (>1.5x 20-period average) reduces false signals
# Session filter (08-20 UTC) avoids low-liquidity hours
# Target: 15-30 trades/year per symbol with disciplined entries
name = "1h_RSI_Supertrend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Supertrend for trend bias
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    # Calculate ATR for Supertrend
    hl = df_4h['high'] - df_4h['low']
    hc = np.abs(df_4h['high'] - df_4h['close'].shift())
    lc = np.abs(df_4h['low'] - df_4h['close'].shift())
    tr = np.maximum(hl, np.maximum(hc, lc))
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (df_4h['high'] + df_4h['low']) / 2
    upper = hl2 + (3.0 * atr)
    lower = hl2 - (3.0 * atr)
    
    supertrend = np.full_like(df_4h['close'], np.nan, dtype=float)
    direction = np.full_like(df_4h['close'], np.nan, dtype=float)
    
    for i in range(10, len(df_4h)):
        if np.isnan(atr[i]) or np.isnan(upper[i]) or np.isnan(lower[i]):
            continue
        if i == 10:
            supertrend[i] = lower[i]
            direction[i] = 1
        else:
            if close_4h := df_4h['close'].iloc[i]:
                if supertrend[i-1] == upper[i-1]:
                    supertrend[i] = lower[i] if close_4h < lower[i] else upper[i]
                    direction[i] = 1 if supertrend[i] == lower[i] else -1
                else:
                    supertrend[i] = upper[i] if close_4h > upper[i] else lower[i]
                    direction[i] = -1 if supertrend[i] == upper[i] else 1
    
    # Align Supertrend direction to 1h
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(rsi[i]) or np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(volume_ma[i]) or not session_filter[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) + uptrend (Supertrend up) + volume filter
            if (rsi[i] < 30 and supertrend_direction_aligned[i] == 1 and volume_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI > 70 (overbought) + downtrend (Supertrend down) + volume filter
            elif (rsi[i] > 70 and supertrend_direction_aligned[i] == -1 and volume_filter[i]):
                signals[i] = -0.20
                position = -1
                
        elif position == 1:
            # Long: exit if RSI > 50 (mean reversion complete) or trend changes
            if (rsi[i] > 50) or (supertrend_direction_aligned[i] == -1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
                
        elif position == -1:
            # Short: exit if RSI < 50 (mean reversion complete) or trend changes
            if (rsi[i] < 50) or (supertrend_direction_aligned[i] == 1):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals