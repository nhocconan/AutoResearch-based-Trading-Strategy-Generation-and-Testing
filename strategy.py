#!/usr/bin/env python3
"""
1h_rsi_divergence_4h1d_momentum_v1
Hypothesis: RSI divergence (RSI vs price) on 1h timeframe filtered by 4h momentum and 1d trend.
Long: Bullish RSI divergence (higher low in RSI, lower low in price) with 4h RSI > 50 and price above 1d EMA50.
Short: Bearish RSI divergence (lower high in RSI, higher high in price) with 4h RSI < 50 and price below 1d EMA50.
Designed for 15-25 trades/year on 1h timeframe with high-conviction signals that work in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_rsi_divergence_4h1d_momentum_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    
    # 4h data for momentum filter (RSI)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h, prepend=close_4h[0])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    avg_gain_4h = pd.Series(gain_4h).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss_4h = pd.Series(loss_4h).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs_4h = avg_gain_4h / (avg_loss_4h + 1e-10)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # 1d data for trend filter (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1h RSI for divergence detection
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, min_periods=14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if data not available
        if (np.isnan(rsi_4h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or
            np.isnan(rsi[i]) or np.isnan(rsi[i-1]) or np.isnan(close[i]) or np.isnan(close[i-1])):
            signals[i] = 0.0
            continue
        
        # RSI divergence detection (requires lookback of at least 5 periods)
        if i >= 5:
            # Bullish divergence: RSI makes higher low, price makes lower low
            bullish_div = (
                rsi[i] > rsi[i-5] and  # RSI higher low
                close[i] < close[i-5]   # Price lower low
            )
            # Bearish divergence: RSI makes lower high, price makes higher high
            bearish_div = (
                rsi[i] < rsi[i-5] and  # RSI lower high
                close[i] > close[i-5]   # Price higher high
            )
        else:
            bullish_div = False
            bearish_div = False
        
        # Momentum and trend filters
        bullish_momentum = rsi_4h_aligned[i] > 50
        bearish_momentum = rsi_4h_aligned[i] < 50
        above_1d_ema50 = close[i] > ema50_1d_aligned[i]
        below_1d_ema50 = close[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: bearish divergence or momentum turns bearish
            if bearish_div or not bullish_momentum:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: bullish divergence or momentum turns bullish
            if bullish_div or not bearish_momentum:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat, look for entry
            # Long: bullish divergence with bullish momentum and uptrend
            if bullish_div and bullish_momentum and above_1d_ema50:
                position = 1
                signals[i] = 0.20
            # Short: bearish divergence with bearish momentum and downtrend
            elif bearish_div and bearish_momentum and below_1d_ema50:
                position = -1
                signals[i] = -0.20
    
    return signals