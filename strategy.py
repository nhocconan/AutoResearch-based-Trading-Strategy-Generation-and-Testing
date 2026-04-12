#!/usr/bin/env python3
# 1h_4d_rsi_momentum_breakout
# Hypothesis: 1-hour RSI momentum breakout with 4-hour trend filter and volume confirmation
# Uses 4h RSI(14) for trend direction (above 50 = bullish, below 50 = bearish)
# 1h RSI(14) crosses above/below 50/70/30 with volume confirmation for entry
# Designed for 15-30 trades/year (60-120 over 4 years) to minimize fee drag
# Works in bull/bear by aligning with higher timeframe momentum

name = "1h_4d_rsi_momentum_breakout"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # 4h RSI(14) for trend filter
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1h RSI(14) for entry signals
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from 4h RSI
        bullish_trend = rsi_4h_aligned[i] > 50
        bearish_trend = rsi_4h_aligned[i] < 50
        
        # Long entry: 1h RSI crosses above 50 in bullish trend with volume
        if (rsi[i] > 50 and rsi[i-1] <= 50 and bullish_trend and 
            vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.20
        # Short entry: 1h RSI crosses below 50 in bearish trend with volume
        elif (rsi[i] < 50 and rsi[i-1] >= 50 and bearish_trend and 
              vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit: reverse signal or RSI reaches extreme levels
        elif position == 1 and (rsi[i] >= 70 or rsi[i] < 50):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi[i] <= 30 or rsi[i] > 50):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals