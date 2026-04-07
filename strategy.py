#!/usr/bin/env python3
"""
1h_4h1d_trend_volume_v1
Hypothesis: Use 4h trend direction (EMA crossover) and 1d trend (price above/below EMA200) as filters,
with 1h RSI pullback entries. Enter long when 4h EMA(21)>EMA(50), 1d close>EMA200, and 1h RSI(14)<30.
Enter short when 4h EMA(21)<EMA(50), 1d close<EMA200, and 1h RSI(14)>70.
Exit when RSI returns to 50 (mean reversion within trend).
Designed for 15-35 trades/year to avoid fee drag while capturing trend continuations with institutional-grade filters.
Works in bull/bear markets as EMA filters adapt and RSI prevents chasing extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_trend_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(21) and EMA(50)
    close_4h = df_4h['close'].values
    ema21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 4h trend: EMA21 > EMA50 = uptrend, EMA21 < EMA50 = downtrend
    trend_4h_up = ema21_4h_aligned > ema50_4h_aligned
    trend_4h_down = ema21_4h_aligned < ema50_4h_aligned
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA(200)
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # 1d trend: price above/below EMA200
    trend_1d_up = close_1d > ema200_1d
    trend_1d_down = close_1d < ema200_1d
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_200_up := trend_1d_up)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down)
    
    # 1h RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if data not available
        if (np.isnan(ema21_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI returns to 50
            if rsi[i] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: RSI returns to 50
            if rsi[i] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Only enter with trend alignment
            if trend_4h_up[i] and trend_1d_up_aligned[i]:
                # Long: RSI pullback to oversold
                if rsi[i] < 30 and rsi[i-1] >= 30:
                    position = 1
                    signals[i] = 0.20
            elif trend_4h_down[i] and trend_1d_down_aligned[i]:
                # Short: RSI pullback to overbought
                if rsi[i] > 70 and rsi[i-1] <= 70:
                    position = -1
                    signals[i] = -0.20
    
    return signals