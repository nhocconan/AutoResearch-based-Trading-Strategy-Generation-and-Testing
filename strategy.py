#!/usr/bin/env python3
"""
6h_1d_1w_MultiTF_Trend_Momentum_v1
Hypothesis: Combine 1d trend (EMA200), 6h momentum (RSI pullback), and 1w momentum (RSI>50) for high-probability entries.
Works in bull by buying dips above EMA200, works in bear by shorting rallies below EMA200 when weekly momentum turns bearish.
Targets 15-30 trades/year to minimize fee drag. Uses 6h timeframe for optimal balance of signal quality and frequency.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_1w_MultiTF_Trend_Momentum_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Weekly data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate daily EMA200
    close_1d = pd.Series(df_1d['close'])
    ema200_1d = close_1d.ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate weekly RSI(14)
    close_1w = pd.Series(df_1w['close'])
    delta_1w = close_1w.diff()
    gain_1w = delta_1w.where(delta_1w > 0, 0)
    loss_1w = (-delta_1w).where(delta_1w < 0, 0)
    avg_gain_1w = gain_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss_1w = loss_1w.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs_1w = avg_gain_1w / avg_loss_1w.replace(0, np.nan)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    rsi_1w = rsi_1w.fillna(50).values  # neutral when undefined
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    # Calculate 6h RSI(14) for entry timing
    close_series = pd.Series(close)
    delta = close_series.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any data invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(rsi_1w_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine trend from daily EMA200
        above_ema200 = close[i] > ema200_1d_aligned[i]
        below_ema200 = close[i] < ema200_1d_aligned[i]
        
        # Weekly momentum filter
        weekly_bullish = rsi_1w_aligned[i] > 50
        weekly_bearish = rsi_1w_aligned[i] < 50
        
        # 6h RSI for entry timing (pullbacks in trend direction)
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        # Long setup: price above daily EMA200, weekly bullish, and 6h RSI oversold
        long_setup = above_ema200 and weekly_bullish and rsi_oversold
        # Short setup: price below daily EMA200, weekly bearish, and 6h RSI overbought
        short_setup = below_ema200 and weekly_bearish and rsi_overbought
        
        # Exit when RSI returns to neutral zone (40-60) or trend fails
        long_exit = (rsi[i] >= 40) or (not above_ema200)
        short_exit = (rsi[i] <= 60) or (not below_ema200)
        
        # Signal logic
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals