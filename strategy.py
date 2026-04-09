#!/usr/bin/env python3
# mtf_1h_ema_rsi_pullback_4h1d_v1
# Hypothesis: On 1h timeframe, enter pullbacks in the direction of the 4h/1d trend.
# 4h EMA(50) and 1d EMA(200) define the trend (both must agree).
# 1h RSI(14) pullback to 30-40 (long) or 60-70 (short) provides entry during uptrend/downtrend.
# 1h volume > 1.5x 20-period average confirms momentum.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Discrete sizing (0.0, ±0.20) minimizes fee churn. Target: 15-35 trades/year.
# Works in bull/bear: trend filter prevents counter-trend trades, volume ensures follow-through.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1h_ema_rsi_pullback_4h1d_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h EMA(50) for intermediate trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA(200) for long-term trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 1h RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 1h volume MA(20)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(140, n):  # Warmup for 1d EMA(200) and other indicators
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_200_1d_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI > 70 (overbought) OR trend turns bearish (4h EMA < 1d EMA)
            if rsi_values[i] > 70 or ema_50_4h_aligned[i] < ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI < 30 (oversold) OR trend turns bullish (4h EMA > 1d EMA)
            if rsi_values[i] < 30 or ema_50_4h_aligned[i] > ema_200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            if volume_confirmed and in_session:
                # Long entry: RSI pullback to 30-40 in uptrend (4h EMA > 1d EMA)
                if 30 <= rsi_values[i] <= 40 and ema_50_4h_aligned[i] > ema_200_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.20
                # Short entry: RSI pullback to 60-70 in downtrend (4h EMA < 1d EMA)
                elif 60 <= rsi_values[i] <= 70 and ema_50_4h_aligned[i] < ema_200_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.20
    
    return signals