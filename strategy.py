#!/usr/bin/env python3
# 4h_PullbackToEMA_Trend_Reversal_Signal
# Hypothesis: Buy pullbacks to 12h EMA50 in uptrends, sell pullbacks to 12h EMA50 in downtrends.
# Uses 12h EMA50 as dynamic support/resistance and 4h RSI for oversold/overbought entries.
# Designed to work in both bull and bear markets by trading reversals with trend bias.
# Targets ~30 trades/year to minimize fee drag.

name = "4h_PullbackToEMA_Trend_Reversal_Signal"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema50_12h
    trend_12h_down = close_12h < ema50_12h
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up.astype(float))
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down.astype(float))
    
    # 4h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: pullback to 12h EMA50 in uptrend with RSI oversold
            if (low[i] <= ema50_12h[i] * 1.005 and  # within 0.5% of EMA
                trend_12h_up_aligned[i] > 0.5 and
                rsi[i] < 30):
                signals[i] = 0.25
                position = 1
            # Short: pullback to 12h EMA50 in downtrend with RSI overbought
            elif (high[i] >= ema50_12h[i] * 0.995 and  # within 0.5% of EMA
                  trend_12h_down_aligned[i] > 0.5 and
                  rsi[i] > 70):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: RSI overbought or price breaks below 12h EMA50
            if (rsi[i] > 70 or
                close[i] < ema50_12h[i] * 0.995):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: RSI oversold or price breaks above 12h EMA50
            if (rsi[i] < 30 or
                close[i] > ema50_12h[i] * 1.005):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals