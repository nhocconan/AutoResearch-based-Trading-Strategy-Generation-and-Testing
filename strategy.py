#!/usr/bin/env python3
"""
1h_1d_rsi_volatility_breakout
Uses daily RSI extremes and volatility contraction (ATR ratio) to detect exhaustion points.
Enters on 1h breakouts in the direction of the extreme with volume confirmation.
Designed for low trade frequency (target: 15-30 trades/year) to minimize fee drift.
Works in both bull and bear markets by capturing mean reversion after overextension.
"""

name = "1h_1d_rsi_volatility_breakout"
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
    
    # Get daily data for RSI and ATR
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Daily RSI (14)
    rsi_length = 14
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length, min_periods=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length, min_periods=rsi_length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # neutral when undefined
    
    # Daily ATR (14) for volatility measurement
    atr_length = 14
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(abs(high_1d - pd.Series(close_1d).shift()))
    tr3 = pd.Series(abs(low_1d - pd.Series(close_1d).shift()))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=atr_length, min_periods=atr_length).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility contraction/expansion)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr / np.where(atr_ma == 0, np.nan, atr_ma)
    
    # Align daily indicators to 1h
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # 1h volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: daily RSI oversold (<30) + volatility contraction (ATR ratio < 0.8) 
        # + 1h breakout above recent high + volume
        if (rsi_aligned[i] < 30 and atr_ratio_aligned[i] < 0.8 and 
            close[i] > np.max(high[max(0, i-20):i]) and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.20
        # Short entry: daily RSI overbought (>70) + volatility contraction (ATR ratio < 0.8)
        # + 1h breakout below recent low + volume
        elif (rsi_aligned[i] > 70 and atr_ratio_aligned[i] < 0.8 and 
              close[i] < np.min(low[max(0, i-20):i]) and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.20
        # Exit conditions: RSI returns to neutral zone (40-60) or volatility expands
        elif position == 1 and (rsi_aligned[i] > 50 or atr_ratio_aligned[i] > 1.2):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (rsi_aligned[i] < 50 or atr_ratio_aligned[i] > 1.2):
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