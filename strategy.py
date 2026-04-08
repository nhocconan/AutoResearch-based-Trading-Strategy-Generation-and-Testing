#!/usr/bin/env python3
# 6h_12h_1d_momentum_reversal_v1
# Hypothesis: Mean reversion on 6h with 12h/1d trend filter captures pullbacks in trending markets. Uses RSI(6) on 6h for overbought/oversold, 12h EMA50 for trend direction, and 1d volume spike for confirmation. Designed to work in both bull and bear markets by trading pullbacks, not chasing momentum.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_1d_momentum_reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 6h RSI(6)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/6, min_periods=6, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/6, min_periods=6, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.values
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate 1d volume spike: volume > 2x average of last 20 days
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = vol_1d > vol_ma_1d * 2.0
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi[i]) or np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (mean reversion complete)
            if rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (mean reversion complete)
            if rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), above 12h EMA50 (uptrend), with volume spike
            if rsi[i] < 30 and close[i] > ema50_12h_aligned[i] and vol_spike_1d_aligned[i] > 0.5:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought), below 12h EMA50 (downtrend), with volume spike
            elif rsi[i] > 70 and close[i] < ema50_12h_aligned[i] and vol_spike_1d_aligned[i] > 0.5:
                position = -1
                signals[i] = -0.25
    
    return signals