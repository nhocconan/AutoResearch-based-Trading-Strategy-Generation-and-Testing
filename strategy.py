# 12h_Combined_Strategy_v1
# Hypothesis: Combines weekly trend (above/below SMA50), daily RSI mean reversion, and 12h momentum for high-probability entries.
# Weekly trend provides directional bias, daily RSI identifies overbought/oversold conditions within the trend, and 12h momentum confirms entry timing.
# Designed to work in both bull and bear markets by following the higher timeframe trend while using lower timeframe for entry precision.
# Target: 15-25 trades/year per symbol with disciplined risk management.

name = "12h_Combined_Strategy_v1"
timeframe = "12h"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate weekly SMA50 for trend filter
    sma_50_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 50:
        sma_50_1w[49] = np.mean(close_1w[0:50])
        for i in range(50, len(close_1w)):
            sma_50_1w[i] = (sma_50_1w[i-1] * 49 + close_1w[i]) / 50
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[0:14])
        avg_loss[13] = np.mean(loss[0:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full_like(close_1d, np.nan)
    valid_avg = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss != 0)
    rs[valid_avg] = avg_gain[valid_avg] / avg_loss[valid_avg]
    
    rsi = np.full_like(close_1d, np.nan)
    rsi[valid_avg] = 100 - (100 / (1 + rs[valid_avg]))
    
    # Align weekly and daily indicators to 12h timeframe
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate 12h momentum (rate of change over 3 periods)
    roc = np.full_like(close, np.nan)
    if len(close) >= 3:
        roc[2:] = (close[2:] - close[:-2]) / close[:-2] * 100
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 14, 2)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(sma_50_1w_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(roc[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend: above SMA50 = uptrend, below = downtrend
        weekly_uptrend = close[i] > sma_50_1w_aligned[i]
        weekly_downtrend = close[i] < sma_50_1w_aligned[i]
        
        if position == 0:
            # Enter long: Weekly uptrend, RSI oversold (<30), and positive momentum
            if weekly_uptrend and rsi_aligned[i] < 30 and roc[i] > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Weekly downtrend, RSI overbought (>70), and negative momentum
            elif weekly_downtrend and rsi_aligned[i] > 70 and roc[i] < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Weekly trend turns down OR RSI overbought (>70)
            if not weekly_uptrend or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Weekly trend turns up OR RSI oversold (<30)
            if not weekly_downtrend or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals