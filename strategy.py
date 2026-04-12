#!/usr/bin/env python3
"""
4h_1d_rsi_volatility_breakout_v1
Hypothesis: 4-hour strategy using daily RSI for momentum and 4-hour Bollinger Bands for volatility breakout.
Works in bull/bear by requiring RSI > 50 for longs and RSI < 50 for shorts, with volume confirmation.
Targets 20-40 trades/year to minimize fee drag while capturing strong momentum moves.
"""

name = "4h_1d_rsi_volatility_breakout_v1"
timeframe = "4h"
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
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 4-hour Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std_dev = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper_band = sma + (std_dev * bb_std)
    lower_band = sma - (std_dev * bb_std)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(sma[i])):
            signals[i] = 0.0
            continue
        
        # Long entry: RSI > 50 (bullish momentum) AND price breaks above upper BB with volume
        if (rsi_1d_aligned[i] > 50 and close[i] > upper_band[i] and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        # Short entry: RSI < 50 (bearish momentum) AND price breaks below lower BB with volume
        elif (rsi_1d_aligned[i] < 50 and close[i] < lower_band[i] and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: price returns to middle Bollinger Band
        elif position == 1 and close[i] < sma[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > sma[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals