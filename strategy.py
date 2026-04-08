#!/usr/bin/env python3
# 6h_1d_1w_rsi_momentum_reversal_v1
# Hypothesis: Uses RSI momentum reversal on 6h timeframe with 1d trend filter and 1w regime filter.
# In bull markets (1w RSI > 50), look for 6h RSI < 30 reversals; in bear markets (1w RSI < 50),
# look for 6h RSI > 70 reversals. Requires 1d EMA(50) alignment to avoid counter-trend trades.
# Designed for 6h timeframe targeting 20-40 trades/year by requiring multi-timeframe alignment
# and extreme RSI readings. Works in bull markets (buying dips) and bear markets (selling rallies).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_rsi_momentum_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 1w RSI(14) for regime filter
    close_1w = df_1w['close'].values
    delta_1w = pd.Series(close_1w).diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = -delta_1w.where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w_values = rsi_1w.fillna(50).values
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    
    # Calculate 6h RSI(14) for entry signals
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(rsi_values[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 70 (overbought) or trend breaks
            if rsi_values[i] >= 70 or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 30 (oversold) or trend breaks
            if rsi_values[i] <= 30 or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Determine market regime from 1w RSI
            bull_regime = rsi_1w_aligned[i] > 50
            bear_regime = rsi_1w_aligned[i] < 50
            
            # Long entry: RSI oversold in bull regime or any oversold with strong uptrend
            if ((bull_regime and rsi_values[i] < 30) or 
                (rsi_values[i] < 25 and close[i] > ema_1d_aligned[i])):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought in bear regime or any overbought with strong downtrend
            elif ((bear_regime and rsi_values[i] > 70) or 
                  (rsi_values[i] > 75 and close[i] < ema_1d_aligned[i])):
                position = -1
                signals[i] = -0.25
    
    return signals