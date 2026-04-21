#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-week RSI extreme with 1-day EMA50 trend filter and volume confirmation.
In uptrend (price > 1d EMA50), buy when 1w RSI < 30 (oversold); in downtrend (price < 1d EMA50), sell when 1w RSI > 70 (overbought).
Volume must exceed 1.5x 20-period average to confirm momentum shift. Exit on RSI reversion to 50.
Designed for 12-30 trades/year (48-120 total over 4 years) to minimize fee fade while capturing mean reversion in trends.
Works in bull via buying dips in uptrend and in bear via selling rallies in downtrend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w and 1d data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 14 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w RSI and 1d EMA50 to 12h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_1w_aligned[i]) or np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_val = rsi_1w_aligned[i]
        ema_trend = ema_50_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: 1w RSI oversold (<30) + uptrend + volume confirmation
            if (rsi_val < 30 and 
                price_close > ema_trend and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: 1w RSI overbought (>70) + downtrend + volume confirmation
            elif (rsi_val > 70 and 
                  price_close < ema_trend and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: RSI returns to neutral zone (40-60)
            exit_signal = False
            
            if position == 1 and rsi_val >= 40:
                exit_signal = True
            elif position == -1 and rsi_val <= 60:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_RSI1w_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0