#!/usr/bin/env python3
"""
6h_1d_RSI_Divergence_Trend_v1
Concept: 60-period RSI on 1d timeframe for trend direction, with 14-period RSI on 6t for overbought/oversold entries.
- Long: 1d RSI(60) > 50 (bullish trend) AND 6h RSI(14) crosses above 30 (oversold bounce)
- Short: 1d RSI(60) < 50 (bearish trend) AND 6h RSI(14) crosses below 70 (overbought rejection)
- Exit: Opposite RSI(14) cross (70 for long exit, 30 for short exit) OR trend reversal (1d RSI crosses 50)
- Position sizing: 0.25
- Target: 20-40 trades/year (80-160 total over 4 years)
- Works in bull/bear: 1d RSI defines trend, 6h RSI captures mean-reversion within trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_RSI_Divergence_Trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 70:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 6h: Price and RSI(14) for entry/exit ===
    close = prices['close'].values
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / np.where(avg_loss > 0, avg_loss, np.nan)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # === 1d: RSI(60) for trend filter ===
    close_1d = df_1d['close'].values
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/60, adjust=False, min_periods=60).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/60, adjust=False, min_periods=60).mean().values
    rs_1d = avg_gain_1d / np.where(avg_loss_1d > 0, avg_loss_1d, np.nan)
    rsi_60_1d = 100 - (100 / (1 + rs_1d))
    rsi_60_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_60_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for RSI(60)
    
    for i in range(start_idx, n):
        # Get values
        rsi_14_val = rsi_14[i]
        rsi_60_1d_val = rsi_60_1d_aligned[i]
        
        # Skip if any value is NaN
        if np.isnan(rsi_14_val) or np.isnan(rsi_60_1d_val):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish trend (1d RSI > 50) AND 6h RSI crosses above 30 (oversold bounce)
            if rsi_60_1d_val > 50 and rsi_14_val > 30 and rsi_14[i-1] <= 30:
                signals[i] = 0.25
                position = 1
            # Short: Bearish trend (1d RSI < 50) AND 6h RSI crosses below 70 (overbought rejection)
            elif rsi_60_1d_val < 50 and rsi_14_val < 70 and rsi_14[i-1] >= 70:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI crosses below 70 (overbought) OR trend turns bearish (1d RSI < 50)
            if rsi_14_val < 70 and rsi_14[i-1] >= 70:
                signals[i] = 0.0
                position = 0
            elif rsi_60_1d_val < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI crosses above 30 (oversold) OR trend turns bullish (1d RSI > 50)
            if rsi_14_val > 30 and rsi_14[i-1] <= 30:
                signals[i] = 0.0
                position = 0
            elif rsi_60_1d_val > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals