#!/usr/bin/env python3
"""
1d_Weekly_RSI_Momentum_Conservative
Hypothesis: Uses weekly RSI to capture long-term momentum with conservative entry/exit.
Long when weekly RSI > 55 and price above 50-day SMA, short when weekly RSI < 45 and price below 50-day SMA.
Exit when RSI crosses back to neutral zone (45-55). Uses volume confirmation to avoid false signals.
Target: 10-20 trades/year to minimize fee drag while capturing major trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI calculation
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly RSI (14-period)
    weekly_close = df_weekly['close'].values
    delta = np.diff(weekly_close, prepend=weekly_close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full_like(weekly_close, np.nan)
    avg_loss = np.full_like(weekly_close, np.nan)
    
    for i in range(14, len(weekly_close)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    weekly_rsi = 100 - (100 / (1 + rs))
    
    # Align weekly RSI to daily timeframe
    weekly_rsi_aligned = align_htf_to_ltf(prices, df_weekly, weekly_rsi)
    
    # Daily 50-day SMA for trend filter
    sma50 = np.full(n, np.nan)
    for i in range(50, n):
        sma50[i] = np.mean(close[i-50:i])
    
    # Volume confirmation: current volume > 1.5 x 20-day average
    vol_ma20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma20[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_rsi_aligned[i]) or np.isnan(sma50[i]) or 
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly RSI > 55, price above SMA50, volume confirmation
            if (weekly_rsi_aligned[i] > 55 and close[i] > sma50[i] and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: weekly RSI < 45, price below SMA50, volume confirmation
            elif (weekly_rsi_aligned[i] < 45 and close[i] < sma50[i] and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: weekly RSI drops below 45 or price below SMA50
            if (weekly_rsi_aligned[i] < 45 or close[i] < sma50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly RSI rises above 55 or price above SMA50
            if (weekly_rsi_aligned[i] > 55 or close[i] > sma50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Weekly_RSI_Momentum_Conservative"
timeframe = "1d"
leverage = 1.0