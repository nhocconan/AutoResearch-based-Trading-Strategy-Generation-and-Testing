#!/usr/bin/env python3

name = "1d_RSI_MeanReversion_With_TrendFilter"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate daily RSI(14) for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (equivalent to EMA with alpha=1/14)
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, 50), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 5  # Prevent overtrading (approx 5 days for 1d)
    
    start_idx = 14  # Warmup for RSI
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine weekly trend direction
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)
        trend_up = weekly_close_aligned[i] > ema_20_1w_aligned[i]
        trend_down = weekly_close_aligned[i] < ema_20_1w_aligned[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: RSI oversold in weekly uptrend
            if rsi[i] < 30 and trend_up:
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: RSI overbought in weekly downtrend
            elif rsi[i] > 70 and trend_down:
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: RSI returns to neutral or trend changes
            if rsi[i] > 50 or not trend_up:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI returns to neutral or trend changes
            if rsi[i] < 50 or not trend_down:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals