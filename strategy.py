#!/usr/bin/env python3
# 1d_1w_rsi_mean_reversion_v1
# Hypothesis: Weekly RSI extreme reversals on daily timeframe. In strong trends (bull/bear),
# weekly RSI >70 or <30 indicates overextension, mean-reverting on daily timeframe.
# Uses 1-day RSI(14) for entry timing and 1-week RSI(14) for regime filter.
# Works in both bull and bear markets by fading extremes in the higher timeframe momentum.
# Low trade frequency: targets 10-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_rsi_mean_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Calculate daily RSI(14) for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi_daily = np.full(n, 50.0)
    for i in range(14, n):
        if avg_loss[i] == 0:
            rsi_daily[i] = 100
        else:
            rs = avg_gain[i] / avg_loss[i]
            rsi_daily[i] = 100 - (100 / (1 + rs))
    
    # Get weekly data for RSI filter (using 1w timeframe)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly RSI(14)
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = np.full(len(close_1w), np.nan)
    avg_loss_1w = np.full(len(close_1w), np.nan)
    for i in range(14, len(close_1w)):
        if i == 14:
            avg_gain_1w[i] = np.mean(gain_1w[1:15])
            avg_loss_1w[i] = np.mean(loss_1w[1:15])
        else:
            avg_gain_1w[i] = (avg_gain_1w[i-1] * 13 + gain_1w[i]) / 14
            avg_loss_1w[i] = (avg_loss_1w[i-1] * 13 + loss_1w[i]) / 14
    
    rsi_1w = np.full(len(close_1w), 50.0)
    for i in range(14, len(close_1w)):
        if avg_loss_1w[i] == 0:
            rsi_1w[i] = 100
        else:
            rs = avg_gain_1w[i] / avg_loss_1w[i]
            rsi_1w[i] = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe (wait for weekly bar to close)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):  # Start after enough data for both RSIs
        rsi_d = rsi_daily[i]
        rsi_w = rsi_1w_aligned[i]
        
        if np.isnan(rsi_d) or np.isnan(rsi_w):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions: daily RSI extreme in opposite direction of weekly extreme
        if position == 0:  # Flat - look for new entries
            # Weekly RSI overbought (>70) and daily RSI oversold (<30) -> short
            # Weekly RSI oversold (<30) and daily RSI overbought (>70) -> long
            if rsi_w > 70 and rsi_d < 30:
                position = -1
                signals[i] = -0.25
            elif rsi_w < 30 and rsi_d > 70:
                position = 1
                signals[i] = 0.25
        
        elif position == 1:  # Long position - exit when daily RSI returns to neutral
            if rsi_d > 50:  # Exit when RSI crosses back above 50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when daily RSI returns to neutral
            if rsi_d < 50:  # Exit when RSI crosses back below 50
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals