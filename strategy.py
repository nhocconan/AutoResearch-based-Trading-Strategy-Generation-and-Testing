#!/usr/bin/env python3
"""
1d_weekly_rsi_extreme_reversion_v1
Hypothesis: On 1d timeframe, enter long when RSI(14) crosses below 30 (oversold) with above-average volume and price above 200-day EMA, enter short when RSI(14) crosses above 70 (overbought) with above-average volume and price below 200-day EMA. Exit when RSI crosses back to 50 (mean reversion). Uses weekly RSI(14) trend filter to avoid counter-trend trades. Designed for 7-25 trades/year on 1d timeframe to minimize fee drag while capturing mean reversion in both bull and bear markets.
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate daily RSI (14-period)
    if len(close) < 14:
        return np.zeros(n)
    
    # RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    
    # Avoid division by zero
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 200-day EMA for trend filter
    ema_200 = pd.Series(close).ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate weekly RSI for trend filter (avoid counter-trend trades)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly RSI calculation
    delta_1w = np.diff(close_1w, prepend=close_1w[0])
    gain_1w = np.where(delta_1w > 0, delta_1w, 0)
    loss_1w = np.where(delta_1w < 0, -delta_1w, 0)
    
    avg_gain_1w = pd.Series(gain_1w).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss_1w = pd.Series(loss_1w).ewm(alpha=1/14, adjust=False).mean().values
    
    rs_1w = np.divide(avg_gain_1w, avg_loss_1w, out=np.zeros_like(avg_gain_1w), where=avg_loss_1w!=0)
    rsi_1w = 100 - (100 / (1 + rs_1w))
    
    # Align weekly RSI to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if data not available
        if (np.isnan(rsi[i]) or np.isnan(rsi_1w_aligned[i]) or np.isnan(ema_200[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: RSI crosses back to 50 (mean reversion)
            if rsi[i] > 50 and rsi[i-1] <= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI crosses back to 50 (mean reversion)
            if rsi[i] < 50 and rsi[i-1] >= 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Long: RSI crosses below 30 with price above EMA200 and weekly RSI bullish (>50)
                if rsi[i] < 30 and rsi[i-1] >= 30 and close[i] > ema_200[i] and rsi_1w_aligned[i] > 50:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI crosses above 70 with price below EMA200 and weekly RSI bearish (<50)
                elif rsi[i] > 70 and rsi[i-1] <= 70 and close[i] < ema_200[i] and rsi_1w_aligned[i] < 50:
                    position = -1
                    signals[i] = -0.25
    
    return signals