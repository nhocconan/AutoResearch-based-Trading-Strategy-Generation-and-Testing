#!/usr/bin/env python3
name = "1h_1dTrend_4hMomentum_Session"
timeframe = "1h"
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
    
    # Load daily data for trend filter (1d)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 4h data for momentum signal
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Daily EMA50 for trend filter (trend = price > EMA50)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h RSI for momentum
    delta = pd.Series(df_4h['close']).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi_14_4h = (100 - (100 / (1 + rs))).values
    rsi_14_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_14_4h)
    
    # 1h EMA21 for entry timing
    ema_21_1h = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Pre-compute session hours (UTC 8-20)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi_14_4h_aligned[i]) or 
            np.isnan(ema_21_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 0:
            # Long: 1d uptrend (price > EMA50) + 4h RSI > 50 + price > EMA21
            if in_session and close[i] > ema_50_1d_aligned[i] and rsi_14_4h_aligned[i] > 50 and close[i] > ema_21_1h[i]:
                signals[i] = 0.20
                position = 1
            # Short: 1d downtrend (price < EMA50) + 4h RSI < 50 + price < EMA21
            elif in_session and close[i] < ema_50_1d_aligned[i] and rsi_14_4h_aligned[i] < 50 and close[i] < ema_21_1h[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: 1d trend reversal or price < EMA21
            if close[i] < ema_50_1d_aligned[i] or close[i] < ema_21_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: 1d trend reversal or price > EMA21
            if close[i] > ema_50_1d_aligned[i] or close[i] > ema_21_1h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h trend following with 1d EMA50 trend filter and 4h RSI momentum
# - Uses 1d EMA50 to determine market trend (bullish if price > EMA50, bearish if price < EMA50)
# - Uses 4h RSI(14) for momentum confirmation (>50 bullish, <50 bearish)
# - Uses 1h EMA21 for entry timing and exit signals
# - Session filter (08-20 UTC) to avoid low-volume Asian session noise
# - Position size 0.20 to manage drawdown and reduce fee churn
# - Works in both bull and bear markets by following the 1d trend
# - Multi-timeframe alignment ensures no look-ahead bias
# - Target: 15-30 trades/year to stay within fee limits
# - Simple, robust logic with clear exit conditions