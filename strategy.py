#!/usr/bin/env python3
"""
1d_1w_kama_rsi_v1
Hypothesis: On 1d timeframe, KAMA trend direction combined with RSI extremes and weekly trend alignment captures trend continuation moves. Weekly trend filter avoids counter-trend trades in ranging markets. KAMA adapts to market noise, reducing false signals during choppy periods.
- Long: KAMA upward (price > KAMA) + RSI < 30 (oversold) + weekly uptrend
- Short: KAMA downward (price < KAMA) + RSI > 70 (overbought) + weekly downtrend
- Exit: Opposite KAMA cross or weekly trend reversal
- Position sizing: 0.25 long, -0.25 short
Designed to work in both bull and bear markets by using adaptive trend (KAMA) and mean reversion (RSI) with weekly trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_kama_rsi_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Calculate KAMA (adaptive moving average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)  # Will fix below
    
    # Proper ER calculation
    dir = np.abs(np.diff(close, n=10, prepend=close[:10]))  # 10-period direction
    vol = np.sum(np.abs(np.diff(close)), axis=0)  # Will fix below
    
    # Simpler approach: use pandas for ER calculation
    close_series = pd.Series(close)
    change = abs(close_series.diff())
    volatility = change.rolling(window=10, min_periods=1).sum()
    direction = abs(close_series - close_series.shift(10))
    er = direction / volatility
    er = er.fillna(0)
    
    # Smoothing constants
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)  # Fill NaN with neutral
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema_20_1w
    trend_1w_down = close_1w < ema_20_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1w trend to 1d
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below KAMA OR weekly trend turns down
            if (close[i] < kama[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price crosses above KAMA OR weekly trend turns up
            if (close[i] > kama[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price > KAMA + RSI < 30 (oversold) + weekly uptrend
            if (close[i] > kama[i]) and (rsi[i] < 30) and trend_1w_up_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: Price < KAMA + RSI > 70 (overbought) + weekly downtrend
            elif (close[i] < kama[i]) and (rsi[i] > 70) and trend_1w_down_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals