#!/usr/bin/env python3
# 6h_1d_rsi_ema_volume_v1
# Hypothesis: RSI(14) extreme + EMA20 trend + volume confirmation on 6h timeframe works in both bull and bear markets by capturing momentum reversals during overextended moves while avoiding chop. 6h limits overtrading; RSI avoids buying strength/selling weakness; EMA filters countertrend; volume confirms conviction.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_ema_volume_v1"
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
    
    # Get 6h data for RSI and EMA calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Get 1d data for trend filter (optional - using same timeframe for simplicity)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate RSI(14) on 6h close
    close_6h = df_6h['close'].values
    delta = np.diff(close_6h, prepend=close_6h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA20 on 6h close
    ema20_6h = pd.Series(close_6h).ewm(span=20, adjust=False).mean().values
    
    # Volume confirmation: 6h volume > 1.5x average of last 20 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > vol_ma * 1.5
    
    # Align 6h indicators to lower timeframe (if needed, but same timeframe here)
    rsi_aligned = align_htf_to_ltf(prices, df_6h, rsi)
    ema20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema20_6h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(rsi_aligned[i]) or np.isnan(ema20_6h_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 (momentum fading)
            if rsi_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 (momentum fading)
            if rsi_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold), price above EMA20, with volume confirmation
            if rsi_aligned[i] < 30 and close[i] > ema20_6h_aligned[i] and vol_confirm[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: RSI > 70 (overbought), price below EMA20, with volume confirmation
            elif rsi_aligned[i] > 70 and close[i] < ema20_6h_aligned[i] and vol_confirm[i]:
                position = -1
                signals[i] = -0.25
    
    return signals