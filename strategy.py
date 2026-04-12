#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h_1w_1d_weekly_bias_daily_trend
# Combines weekly trend bias with daily momentum for 6h entries.
# Uses 1w EMA(20) for long-term trend direction and 1d RSI(14) for momentum timing.
# Long when: 1w EMA(20) upward AND 1d RSI crosses above 30 (oversold bounce).
# Short when: 1w EMA(20) downward AND 1d RSI crosses below 70 (overbought rejection).
# Exit when RSI crosses 50 (mean reversion to center).
# Designed for low trade frequency (15-30 trades/year) with clear trend/momentum filters.
# Works in bull (follow weekly uptrend) and bear (short weekly downtrend) markets.

name = "6h_1w_1d_weekly_bias_daily_trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get weekly data for trend bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for momentum timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend bias
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_prev = np.roll(ema_20_1w, 1)
    ema_20_1w_prev[0] = ema_20_1w[0]
    ema_20_1w_up = ema_20_1w > ema_20_1w_prev
    ema_20_1w_down = ema_20_1w < ema_20_1w_prev
    
    # Daily RSI(14) for momentum timing
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align weekly trend and daily RSI to 6h timeframe
    ema_20_1w_up_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w_up)
    ema_20_1w_down_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w_down)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # RSI cross signals
    rsi_prev = np.roll(rsi_aligned, 1)
    rsi_prev[0] = 50
    rsi_cross_up = (rsi_aligned > 30) & (rsi_prev <= 30)
    rsi_cross_down = (rsi_aligned < 70) & (rsi_prev >= 70)
    rsi_cross_50_up = (rsi_aligned > 50) & (rsi_prev <= 50)
    rsi_cross_50_down = (rsi_aligned < 50) & (rsi_prev >= 50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1w_up_aligned[i]) or np.isnan(ema_20_1w_down_aligned[i]) or 
            np.isnan(rsi_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long: weekly uptrend + RSI crosses above 30 (oversold bounce)
        if ema_20_1w_up_aligned[i] and rsi_cross_up[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: weekly downtrend + RSI crosses below 70 (overbought rejection)
        elif ema_20_1w_down_aligned[i] and rsi_cross_down[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: RSI crosses 50 (mean reversion to center)
        elif position == 1 and rsi_cross_50_down[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and rsi_cross_50_up[i]:
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