#!/usr/bin/env python3
# 1h_MidTrend_Momentum_Filter
# Hypothesis: On 1h timeframe, use 4h EMA trend for direction and 1h RSI momentum for entry timing.
# Enter long when price > 4h EMA50 and RSI(14) crosses above 50; enter short when price < 4h EMA50 and RSI crosses below 50.
# Exit on opposite RSI cross or trend reversal. Uses session filter (08-20 UTC) to avoid low-volume hours.
# Target: 20-40 trades/year to minimize fee drag on 1h timeframe.

name = "1h_MidTrend_Momentum_Filter"
timeframe = "1h"
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
    
    # 4h trend filter (EMA50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema50_4h
    trend_4h_down = close_4h < ema50_4h
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up.astype(float))
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down.astype(float))
    
    # 1h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # RSI cross signals
    rsi_above_50 = rsi > 50
    rsi_below_50 = rsi < 50
    rsi_cross_up = np.zeros(n, dtype=bool)
    rsi_cross_down = np.zeros(n, dtype=bool)
    rsi_cross_up[1:] = (rsi_above_50[1:] & ~rsi_above_50[:-1])
    rsi_cross_down[1:] = (rsi_below_50[1:] & ~rsi_below_50[:-1])
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > 4h EMA50, RSI crosses above 50, in session
            if (close[i] > ema50_4h[i] and
                rsi_cross_up[i] and
                session_mask[i]):
                signals[i] = 0.20
                position = 1
            # Short: price < 4h EMA50, RSI crosses below 50, in session
            elif (close[i] < ema50_4h[i] and
                  rsi_cross_down[i] and
                  session_mask[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit: RSI crosses below 50 or trend turns down
            if (rsi_cross_down[i] or
                trend_4h_up_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit: RSI crosses above 50 or trend turns up
            if (rsi_cross_up[i] or
                trend_4h_down_aligned[i] < 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals