#!/usr/bin/env python3
# 1D_Weekly_Momentum_Pullback
# Hypothesis: On daily timeframe, enter long when weekly close > weekly EMA50 and price pulls back to daily EMA20 with RSI < 40.
# Enter short when weekly close < weekly EMA50 and price pulls back to daily EMA20 with RSI > 60.
# Exit on opposite signal or when price crosses daily EMA50.
# Uses weekly trend filter to avoid counter-trend trades and daily EMA20/RSI for precise entries.
# Targets 10-20 trades/year for low fee decay.

name = "1D_Weekly_Momentum_Pullback"
timeframe = "1d"
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
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA50
    ema50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly EMA50 to daily timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate daily indicators
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = (100 - (100 / (1 + rs))).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema20[i]) or 
            np.isnan(ema50[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        weekly_trend = ema50_1w_aligned[i]
        daily_ema20 = ema20[i]
        daily_ema50 = ema50[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # LONG: Weekly uptrend + price pulls back to daily EMA20 + RSI oversold
            if weekly_close[-1] > weekly_trend and close[i] <= daily_ema20 * 1.01 and rsi_val < 40:
                signals[i] = 0.25
                position = 1
            # SHORT: Weekly downtrend + price pulls back to daily EMA20 + RSI overbought
            elif weekly_close[-1] < weekly_trend and close[i] >= daily_ema20 * 0.99 and rsi_val > 60:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly trend turns down OR price crosses below daily EMA50
            if weekly_close[-1] < weekly_trend or close[i] < daily_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly trend turns up OR price crosses above daily EMA50
            if weekly_close[-1] > weekly_trend or close[i] > daily_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals