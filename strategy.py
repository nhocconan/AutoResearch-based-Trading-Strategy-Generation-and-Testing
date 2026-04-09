#!/usr/bin/env python3
# 1h_4d_rsi_mean_reversion_v1
# Hypothesis: On 1h chart, buy when RSI(14) < 30 and price is above 200-period EMA (long-term uptrend filter),
# sell when RSI(14) > 70 and price is below 200-period EMA (long-term downtrend filter).
# Uses 1d trend filter (EMA50) to avoid counter-trend trades in strong trends.
# Designed for 1h timeframe with ~20-40 trades/year to minimize fee drag.
# Works in bull markets (buy dips in uptrend) and bear markets (sell rallies in downtrend).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_rsi_mean_reversion_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) - Wilder's smoothing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    avg_gain[13] = np.mean(gain[1:14])  # First average of gains
    avg_loss[13] = np.mean(loss[1:14])  # First average of losses
    
    for i in range(14, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate EMA200 for trend filter
    ema200 = np.full(n, np.nan)
    ema200[0] = close[0]
    alpha = 2 / (200 + 1)
    for i in range(1, n):
        ema200[i] = alpha * close[i] + (1 - alpha) * ema200[i-1]
    
    # Load 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA50 on daily
    ema50_1d = np.full(len(close_1d), np.nan)
    ema50_1d[0] = close_1d[0]
    alpha_1d = 2 / (50 + 1)
    for i in range(1, len(close_1d)):
        ema50_1d[i] = alpha_1d * close_1d[i] + (1 - alpha_1d) * ema50_1d[i-1]
    
    # Align 1d EMA50 to 1h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Session filter: 08-20 UTC (avoid low liquidity periods)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(rsi[i]) or np.isnan(ema200[i]) or np.isnan(ema50_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral (50) or price breaks below EMA200
            if rsi[i] >= 50 or close[i] < ema200[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral (50) or price breaks above EMA200
            if rsi[i] <= 50 or close[i] > ema200[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: RSI oversold (<30) and price above EMA200 (uptrend filter)
            # AND 1d trend is up (price > EMA50) to avoid buying in strong downtrend
            if rsi[i] < 30 and close[i] > ema200[i] and close[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Enter short: RSI overbought (>70) and price below EMA200 (downtrend filter)
            # AND 1d trend is down (price < EMA50) to avoid selling in strong uptrend
            elif rsi[i] > 70 and close[i] < ema200[i] and close[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals