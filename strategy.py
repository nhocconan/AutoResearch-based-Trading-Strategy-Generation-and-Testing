#!/usr/bin/env python3
# 1d_1w_rsi_mean_reversion
# Hypothesis: RSI(14) mean reversion on 1d with 1w trend filter and volume confirmation works in both bull and bear markets by buying oversold during uptrends and selling overbought during downtrends. 1d timeframe reduces trade frequency; trend filter avoids countertrend trades; volume confirmation ensures momentum. Target: 15-25 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_mean_reversion"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14) on 1d close
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation: 1d volume > 1.3x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.3
    
    # Align indicators to 1d timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses above 50 (momentum fading) or overbought
            if rsi_aligned[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI crosses below 50 (momentum fading) or oversold
            if rsi_aligned[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), above 1w EMA50 (uptrend), with volume confirmation
            if rsi_aligned[i] < 30 and close[i] > ema50_1w_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought), below 1w EMA50 (downtrend), with volume confirmation
            elif rsi_aligned[i] > 70 and close[i] < ema50_1w_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals