#!/usr/bin/env python3
# 1d_PriceAction_Momentum_WeeklyTrend_Filter
# Hypothesis: Use daily price action with momentum confirmation and weekly trend filter.
# Long when price closes above prior day's high with RSI > 50 and weekly trend up.
# Short when price closes below prior day's low with RSI < 50 and weekly trend down.
# Exit when price reverses back to prior day's close or momentum fades.
# Designed to capture momentum bursts with trend alignment, works in both bull and bear markets.
# Targets 15-25 trades/year to minimize fee drag.

name = "1d_PriceAction_Momentum_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 14-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_ema_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure RSI is stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if np.isnan(rsi[i]) or np.isnan(weekly_ema_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Close above prior day's high, RSI > 50, weekly trend up
            if close[i] > high[i-1] and rsi[i] > 50 and close[i] > weekly_ema_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Close below prior day's low, RSI < 50, weekly trend down
            elif close[i] < low[i-1] and rsi[i] < 50 and close[i] < weekly_ema_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close below prior day's close or RSI < 40
            if close[i] < close[i-1] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close above prior day's close or RSI > 60
            if close[i] > close[i-1] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals