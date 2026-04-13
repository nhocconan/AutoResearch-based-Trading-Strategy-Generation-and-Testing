#!/usr/bin/env python3
"""
1h_4h_1d_Momentum_Pullback_V2
Hypothesis: Combines 1d trend filter (price > SMA200) with 4h momentum (RSI > 55 for long, < 45 for short) and precise 1h entry on pullback to EMA21.
Works in bull markets via momentum continuation and in bear via mean-reversion bounces off EMA21 during strong trends.
Target: 15-37 trades/year on 1h (60-150 total over 4 years).
"""

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Daily SMA200 for trend filter
    sma_200_1d = pd.Series(close_1d).rolling(window=200, min_periods=200).mean()
    trend_up = close_1d > sma_200_1d
    trend_down = close_1d < sma_200_1d
    
    # Get 4h data for momentum
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    # 4h RSI(14)
    delta = pd.Series(close_4h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_long = rsi_14 > 55
    rsi_short = rsi_14 < 45
    
    # Align 1d and 4h signals to 1h
    trend_up_aligned = align_htf_to_ltf(prices, df_1d, trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1d, trend_down)
    rsi_long_aligned = align_htf_to_ltf(prices, df_4h, rsi_long)
    rsi_short_aligned = align_htf_to_ltf(prices, df_4h, rsi_short)
    
    # 1h EMA21 for pullback entry
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.20  # 20% of capital
    
    for i in range(50, n):
        # Skip if not in session or data not ready
        if not session_mask[i] or \
           np.isnan(trend_up_aligned[i]) or \
           np.isnan(trend_down_aligned[i]) or \
           np.isnan(rsi_long_aligned[i]) or \
           np.isnan(rsi_short_aligned[i]) or \
           np.isnan(ema_21[i]):
            signals[i] = 0.0
            continue
        
        # Entry logic
        if trend_up_aligned[i] and rsi_long_aligned[i]:
            # Bullish setup: buy on pullback to EMA21
            if close[i] <= ema_21[i] * 1.002:  # Within 0.2% of EMA21
                if position != 1:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = position_size
            elif position == 1:
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif trend_down_aligned[i] and rsi_short_aligned[i]:
            # Bearish setup: sell on bounce to EMA21
            if close[i] >= ema_21[i] * 0.998:  # Within 0.2% of EMA21
                if position != -1:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = -position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        else:
            # No clear setup - flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_1d_Momentum_Pullback_V2"
timeframe = "1h"
leverage = 1.0