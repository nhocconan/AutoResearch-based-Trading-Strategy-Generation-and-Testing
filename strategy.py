#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1-week RSI extremes with 1-week EMA200 trend filter and volume confirmation.
In uptrend (price > 1w EMA200), buy when 1w RSI < 30 (oversold); in downtrend (price < 1w EMA200), sell when 1w RSI > 70 (overbought).
Volume must exceed 1.5x 20-period average to confirm reversal strength. Exit on RSI returning to neutral zone (40-60).
Designed for 15-25 trades/year (60-100 total over 4 years) to minimize fee decay while capturing mean-reversion moves in both bull and bear markets.
Works in bull markets via oversold bounces and in bear markets via overbought reversals with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for RSI and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 1w EMA200 for trend filter
    ema_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w indicators to 1d timeframe (wait for 1w bar to close)
    rsi_aligned = align_htf_to_ltf(prices, df_1w, rsi)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        rsi_val = rsi_aligned[i]
        ema_trend = ema_200_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price > 1w EMA200 (uptrend) + 1w RSI < 30 (oversold) + volume spike
            if (price_close > ema_trend and 
                rsi_val < 30 and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price < 1w EMA200 (downtrend) + 1w RSI > 70 (overbought) + volume spike
            elif (price_close < ema_trend and 
                  rsi_val > 70 and 
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

name = "1d_RSI14_1wEMA200_Volume"
timeframe = "1d"
leverage = 1.0