#!/usr/bin/env python3

# Hypothesis: 1d timeframe with weekly (1w) RSI momentum filter and daily volume confirmation.
# Uses weekly RSI(14) to filter trend direction (avoids counter-trend trades) and daily volume spike for confirmation.
# Weekly RSI provides stable momentum filter that works in both bull and bear markets by avoiding extremes.
# Target: 20-80 total trades over 4 years (5-20/year) with size 0.25 to minimize fee drag.
# Weekly data changes slowly, reducing whipsaw and improving win rate in ranging markets.

name = "1d_RSI14_1wRSI_Filter_Volume"
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
    volume = prices['volume'].values
    
    # Calculate daily RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get weekly data for RSI(14) trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate weekly RSI(14) trend filter
    delta_w = np.diff(df_1w['close'].values, prepend=df_1w['close'].values[0])
    gain_w = np.where(delta_w > 0, delta_w, 0)
    loss_w = np.where(delta_w < 0, -delta_w, 0)
    
    avg_gain_w = pd.Series(gain_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_w = pd.Series(loss_w).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_w = avg_gain_w / (avg_loss_w + 1e-10)
    rsi_w = 100 - (100 / (1 + rs_w))
    
    rsi_w_aligned = align_htf_to_ltf(prices, df_1w, rsi_w)
    
    # Volume filter: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or np.isnan(rsi_w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: daily RSI > 50 (bullish momentum) + weekly RSI > 50 (bullish weekly trend) + volume spike
            if rsi[i] > 50 and rsi_w_aligned[i] > 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: daily RSI < 50 (bearish momentum) + weekly RSI < 50 (bearish weekly trend) + volume spike
            elif rsi[i] < 50 and rsi_w_aligned[i] < 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: daily RSI < 40 (loss of momentum) or weekly RSI < 40 (trend weakening)
            if rsi[i] < 40 or rsi_w_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: daily RSI > 60 (loss of bearish momentum) or weekly RSI > 60 (trend weakening)
            if rsi[i] > 60 or rsi_w_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals