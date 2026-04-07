#!/usr/bin/env python3
"""
1d_weekly_rsi_extreme_reversion_v1
Hypothesis: On daily timeframe, enter long when weekly RSI(14) < 30 (oversold) with daily RSI < 50 and price above 200-day EMA, enter short when weekly RSI(14) > 70 (overbought) with daily RSI > 50 and price below 200-day EMA. Exit when weekly RSI returns to neutral zone (40-60). Uses weekly RSI to avoid counter-trend trades in strong trends. Designed for 7-25 trades/year (30-100 total) to minimize fee drag while capturing mean reversion in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_rsi_extreme_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    
    # Calculate 200-day EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Calculate daily RSI(14)
    if len(close) < 15:
        return np.zeros(n)
    
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_daily = 100 - (100 / (1 + rs))
    
    # Calculate weekly RSI(14) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 15:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False).mean().values
    
    rs_1w = avg_gain_1w / (avg_loss_1w + 1e-10)
    rsi_weekly = 100 - (100 / (1 + rs_1w))
    
    # Align weekly RSI to daily timeframe
    rsi_weekly_aligned = align_htf_to_ltf(prices, df_1w, rsi_weekly)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(rsi_daily[i]) or np.isnan(rsi_weekly_aligned[i]) or 
            np.isnan(ema_200[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: weekly RSI returns to neutral zone (40-60)
            if rsi_weekly_aligned[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: weekly RSI returns to neutral zone (40-60)
            if rsi_weekly_aligned[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: weekly RSI < 30 (oversold) + daily RSI < 50 + price above EMA200
            if (rsi_weekly_aligned[i] < 30 and rsi_daily[i] < 50 and 
                close[i] > ema_200[i]):
                position = 1
                signals[i] = 0.25
            # Short: weekly RSI > 70 (overbought) + daily RSI > 50 + price below EMA200
            elif (rsi_weekly_aligned[i] > 70 and rsi_daily[i] > 50 and 
                  close[i] < ema_200[i]):
                position = -1
                signals[i] = -0.25
    
    return signals