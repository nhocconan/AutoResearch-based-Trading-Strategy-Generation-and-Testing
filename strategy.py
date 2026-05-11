#!/usr/bin/env python3
# Strategy: 12h_Daily_RSI_Extreme_Reversion
# Hypothesis: Daily RSI extremes (oversold/overbought) on 12h timeframe with volume confirmation
# work in both bull and bear markets as mean-reversion signals. Uses 1d RSI for regime,
# 12h price action for entry, and volume filter to avoid false signals. Target: 15-25 trades/year.
name = "12h_Daily_RSI_Extreme_Reversion"
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
    
    # Daily RSI(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 12h RSI(14) for entry signal
    delta_12h = np.diff(close, prepend=close[0])
    gain_12h = np.where(delta_12h > 0, delta_12h, 0)
    loss_12h = np.where(delta_12h < 0, -delta_12h, 0)
    
    avg_gain_12h = pd.Series(gain_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_12h = pd.Series(loss_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_12h = avg_gain_12h / (avg_loss_12h + 1e-10)
    rsi_12h = 100 - (100 / (1 + rs_12h))
    
    # Volume confirmation: 12h volume > 1.3x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.3 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        if np.isnan(rsi_1d_aligned[i]) or np.isnan(rsi_12h[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 12h RSI < 30 (oversold) AND daily RSI < 50 (bearish bias for mean reversion) AND volume
            if rsi_12h[i] < 30 and rsi_1d_aligned[i] < 50 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: 12h RSI > 70 (overbought) AND daily RSI > 50 (bullish bias for mean reversion) AND volume
            elif rsi_12h[i] > 70 and rsi_1d_aligned[i] > 50 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: 12h RSI > 50 (mean reversion complete) or RSI > 70 (overbought)
            if rsi_12h[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: 12h RSI < 50 (mean reversion complete) or RSI < 30 (oversold)
            if rsi_12h[i] < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals