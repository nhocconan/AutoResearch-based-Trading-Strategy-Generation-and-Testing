#!/usr/bin/env python3
"""
1d_MultiFactor_Trend_Momentum_Volume
Hypothesis: Combine daily trend (EMA34), momentum (RSI>50), and volume (20-period volume spike) to capture sustained moves.
Works in bull markets via trend+momentum and in bear via short signals when trend/momentum reverse. Daily timeframe reduces
noise and trade frequency. Volume confirmation ensures institutional participation. Target 15-25 trades/year.
"""

name = "1d_MultiFactor_Trend_Momentum_Volume"
timeframe = "1d"
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
    
    # === Weekly Trend Filter (EMA34) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily Momentum (RSI 14) ===
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # === Volume Filter (2.0x 20-period EMA) ===
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d[i]) or np.isnan(rsi_values[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price above weekly EMA, RSI>50, volume spike
            if (close[i] > ema34_1d[i] and 
                rsi_values[i] > 50 and 
                volume_ok[i]):
                signals[i] = 0.25
                position = 1
            # Short: price below weekly EMA, RSI<50, volume spike
            elif (close[i] < ema34_1d[i] and 
                  rsi_values[i] < 50 and 
                  volume_ok[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below weekly EMA OR RSI<40
            if close[i] < ema34_1d[i] or rsi_values[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price crosses above weekly EMA OR RSI>60
            if close[i] > ema34_1d[i] or rsi_values[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals