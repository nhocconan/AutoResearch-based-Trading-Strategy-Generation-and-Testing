#!/usr/bin/env python3
# 6h_RSI2_MeanReversion_WeeklyTrend
# Hypothesis: On 6h timeframe, mean-reversion trades using 2-period RSI (RSI2) with weekly trend filter.
# RSI2 < 10 signals oversold conditions for long entries; RSI2 > 90 signals overbought for shorts.
# Weekly trend filter (price above/below 200-period EMA) ensures trades align with higher-timeframe momentum.
# Works in bull markets via pullbacks in uptrend and in bear markets via bounces in downtrend.
# Low trade frequency expected due to strict RSI2 thresholds and trend alignment requirement.

name = "6h_RSI2_MeanReversion_WeeklyTrend"
timeframe = "6h"
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
    
    # Load weekly data ONCE for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Weekly EMA200 for trend filter
    ema_200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # RSI(2) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 2  # Need at least 2 periods for RSI(2)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(rsi_values[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_200_1w_aligned[i]
        downtrend = close[i] < ema_200_1w_aligned[i]
        
        # RSI2 conditions
        rsi_oversold = rsi_values[i] < 10
        rsi_overbought = rsi_values[i] > 90
        
        if position == 0:
            # Long: RSI2 oversold + weekly uptrend
            if rsi_oversold and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: RSI2 overbought + weekly downtrend
            elif rsi_overbought and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI2 crosses above 50 (mean reversion complete) or trend fails
            if rsi_values[i] > 50 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI2 crosses below 50 (mean reversion complete) or trend fails
            if rsi_values[i] < 50 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals