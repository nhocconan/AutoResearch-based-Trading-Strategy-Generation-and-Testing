#!/usr/bin/env python3
"""
1h_RSI2_MeanReversion_4h1dTrend_Filter
Hypothesis: RSI(2) mean reversion (RSI<10) works only when aligned with higher timeframe trend (4h EMA50 & 1d EMA200) to avoid counter-trend trades. 
Uses 1h for precise entry timing, 4h/1d for trend filter. Designed for low trade frequency (target: 15-30/year) to minimize fee drag in 1h timeframe.
Works in bull/bear via trend filter: only long in uptrends, only short in downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 4h data for intermediate trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d data for long-term trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # 4h trend: EMA50
    ema50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1d trend: EMA200
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # RSI(2) calculation
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    avg_loss = loss.ewm(alpha=1/2, adjust=False, min_periods=2).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100).values  # when loss=0, RSI=100
    
    # Session filter: 08:00-20:00 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need enough data for RSI and EMAs
    start_idx = max(2, 50, 200)
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if np.isnan(ema50_4h_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or not in_session[i]:
            signals[i] = 0.0
            continue
        
        ema40_trend = ema50_4h_aligned[i]
        ema1d_trend = ema200_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Long: RSI(2) < 10 (oversold) + uptrend on both 4h and 1d
            if rsi_val < 10 and close[i] > ema40_trend and close[i] > ema1d_trend:
                signals[i] = size
                position = 1
            # Short: RSI(2) > 90 (overbought) + downtrend on both 4h and 1d
            elif rsi_val > 90 and close[i] < ema40_trend and close[i] < ema1d_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI crosses above 50 (mean reversion complete) or trend breaks
            if rsi_val > 50 or close[i] < ema40_trend or close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI crosses below 50 (mean reversion complete) or trend breaks
            if rsi_val < 50 or close[i] > ema40_trend or close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI2_MeanReversion_4h1dTrend_Filter"
timeframe = "1h"
leverage = 1.0