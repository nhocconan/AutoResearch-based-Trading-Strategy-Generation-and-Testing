#!/usr/bin/env python3
# 1d_Weekly_RSI_Reversal_With_Volume_Confirmation
# Hypothesis: Uses weekly RSI extremes (>70 or <30) as reversal signals on the daily timeframe.
# Combines with daily price action (close outside Bollinger Bands) and volume confirmation.
# Designed to work in both bull and bear markets by capturing overextended moves.
# Target: 15-25 trades/year (~60-100 total over 4 years) to stay within optimal trade frequency for 1d.

name = "1d_Weekly_RSI_Reversal_With_Volume_Confirmation"
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
    
    # Weekly RSI (14-period) for overbought/oversold conditions
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate RSI for weekly data
    delta = np.diff(close_1w)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.concatenate([[np.nan], rsi_1w])  # align with index 0
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Daily Bollinger Bands (20, 2)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    
    # Volume confirmation: current volume > 1.3 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(bb_middle[i]) or 
            np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Weekly RSI < 30 (oversold) AND price below lower Bollinger Band AND volume confirmation
            if (rsi_1w_aligned[i] < 30 and 
                close[i] < bb_lower[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Weekly RSI > 70 (overbought) AND price above upper Bollinger Band AND volume confirmation
            elif (rsi_1w_aligned[i] > 70 and 
                  close[i] > bb_upper[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Weekly RSI returns to neutral (40-60) OR price crosses above middle band
            if (rsi_1w_aligned[i] >= 40 and rsi_1w_aligned[i] <= 60) or \
               close[i] > bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Weekly RSI returns to neutral (40-60) OR price crosses below middle band
            if (rsi_1w_aligned[i] >= 40 and rsi_1w_aligned[i] <= 60) or \
               close[i] < bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals