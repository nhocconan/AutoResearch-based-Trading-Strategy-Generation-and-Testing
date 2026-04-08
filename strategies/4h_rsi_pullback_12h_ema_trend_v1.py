#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h RSI Pullback with 12h EMA Trend Filter
# Hypothesis: RSI(14) pullbacks in direction of 12h EMA(50) trend capture mean reversion within trends.
# Uses 12h EMA for trend filter (works in bull/bear) and RSI for precise entries.
# Target: 20-35 trades/year (80-140 total over 4 years) to minimize fee drag.

name = "4h_rsi_pullback_12h_ema_trend_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for EMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h close
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False).mean().values
    
    # Align 12h EMA to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        result[period-1] = np.mean(data[1:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = wilders_smoothing(gain, 14)
    avg_loss = wilders_smoothing(loss, 14)
    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(rsi[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI reaches overbought or trend changes
            if rsi[i] >= 70 or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long
        elif position == -1:  # Short position
            # Exit: RSI reaches oversold or trend changes
            if rsi[i] <= 30 or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short
        else:  # Flat, look for entry
            # RSI pullback in direction of 12h trend
            if close[i] > ema_50_aligned[i]:  # Uptrend
                if rsi[i] <= 40:  # Pullback to buy
                    position = 1
                    signals[i] = 0.25
            else:  # Downtrend
                if rsi[i] >= 60:  # Pullback to sell
                    position = -1
                    signals[i] = -0.25
    
    return signals