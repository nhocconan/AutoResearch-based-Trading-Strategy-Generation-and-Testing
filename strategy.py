#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4-hour RSI mean reversion with 1-day trend filter and session filter.
# RSI(14) extremes (overbought/oversold) signal mean reversion entries, filtered by 1-day EMA(50) trend.
# Only trade during active hours (08-20 UTC) to avoid low-liquidity periods.
# Works in both bull and bear markets by combining mean reversion with trend alignment.
# Target: 15-37 trades/year (60-150 total over 4 years) with strict entry conditions.
name = "1h_RSIMeanRev_1dEMA50_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # RSI(14) on 1h
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/rsi_period, adjust=False, min_periods=rsi_period).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable or outside session
        if (np.isnan(rsi[i]) or np.isnan(ema_50_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: RSI oversold (<30) + price above 1-day EMA50 (uptrend filter)
            if (rsi[i] < 30 and price > ema_50_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: RSI overbought (>70) + price below 1-day EMA50 (downtrend filter)
            elif (rsi[i] > 70 and price < ema_50_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>50) or trend changes
            if rsi[i] > 50 or price < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<50) or trend changes
            if rsi[i] < 50 or price > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals