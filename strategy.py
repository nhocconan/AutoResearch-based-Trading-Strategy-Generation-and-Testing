#!/usr/bin/env python3
"""
6H_ADX_Trend_Strength_Volume_Signal
Hypothesis: Uses ADX to detect trending conditions (ADX > 25) and RSI to determine direction 
(RSI > 50 for long, RSI < 50 for short). Volume confirmation ensures breakout validity. 
Designed for 6h timeframe to capture medium-term trends with low trade frequency (target: 15-35 trades/year).
Works in both bull and bear markets by following trend direction, avoiding false signals in ranging markets.
"""

name = "6H_ADX_Trend_Strength_Volume_Signal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for ADX and RSI calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX components on daily data
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    tr_ma = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_ma
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean() / tr_ma
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean()
    
    # RSI on daily close
    delta = df_1d['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # Convert to numpy arrays
    adx_vals = adx.values
    rsi_vals = rsi.values
    
    # Align to 6h timeframe (wait for daily close)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_vals)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi_vals)
    
    # Volume filter: volume > 1.5x 24-period average on 6h chart
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 24)  # Warmup for ADX/RSI and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(adx_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend strength filter: ADX > 25 indicates trending market
        is_trending = adx_aligned[i] > 25
        
        if position == 0:
            # Long entry: RSI > 50 (bullish momentum) + volume + trending market
            if (rsi_aligned[i] > 50 and 
                volume[i] > vol_threshold[i] and 
                is_trending):
                signals[i] = 0.25
                position = 1
            # Short entry: RSI < 50 (bearish momentum) + volume + trending market
            elif (rsi_aligned[i] < 50 and 
                  volume[i] > vol_threshold[i] and 
                  is_trending):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI < 40 (loss of momentum) or ADX < 20 (trend weakening)
            if (rsi_aligned[i] < 40 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI > 60 (loss of bearish momentum) or ADX < 20 (trend weakening)
            if (rsi_aligned[i] > 60 or adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals