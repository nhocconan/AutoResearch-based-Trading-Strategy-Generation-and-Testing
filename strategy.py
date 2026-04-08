#!/usr/bin/env python3
# 1h_rsi_ema_4h_filter_v1
# Hypothesis: On 1h timeframe, RSI(14) below 30 for long and above 70 for short captures short-term reversals.
# Filter: Only trade in direction of 4h EMA(50) trend to avoid counter-trend trades.
# Entry: Long when RSI < 30 and close > 4h EMA50; Short when RSI > 70 and close < 4h EMA50
# Exit: RSI crosses back above 50 (long) or below 50 (short)
# Position sizing: 0.20 long, -0.20 short
# Uses 4h EMA50 for trend filter, 1h RSI for entry timing. Designed for both bull and bear markets.
# Target: 15-37 trades/year (60-150 total over 4 years) by combining tight RSI extremes with trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_ema_4h_filter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMA50 to 1h
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # RSI(14) calculation on 1h
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema_50_4h_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI < 50
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI < 30 and price > 4h EMA50
            if (rsi[i] < 30) and (close[i] > ema_50_4h_aligned[i]):
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 and price < 4h EMA50
            elif (rsi[i] > 70) and (close[i] < ema_50_4h_aligned[i]):
                position = -1
                signals[i] = -0.20
    
    return signals