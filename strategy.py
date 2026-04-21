#!/usr/bin/env python3
"""
Hypothesis: 12h strategy using 1-day KAMA trend filter with RSI momentum and volume confirmation.
In uptrend (price > 1-day KAMA), buy when RSI crosses above 50 with volume spike; in downtrend (price < 1-day KAMA), 
sell when RSI crosses below 50 with volume spike. Uses volume to confirm momentum strength.
Targets ~15-25 trades/year (60-100 total over 4 years) to avoid fee drag.
Works in bull markets via momentum longs and in bear markets via momentum shorts with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1-day KAMA ( Kaufman Adaptive Moving Average )
    # Efficient Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly: sum of absolute changes over 10 periods
    volatility = np.zeros_like(close_1d)
    for i in range(len(close_1d)):
        if i < 10:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[max(0, i-9):i+1]))) if i > 0 else 0
        else:
            volatility[i] = np.sum(np.abs(np.diff(close_1d[i-9:i+1])))
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Alternative simpler approach: use EMA as trend proxy if KAMA too complex
    # But let's implement proper KAMA with vectorized approach where possible
    # Recalculate using pandas for simplicity and correctness
    close_series = pd.Series(close_1d)
    change = close_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=1).sum()
    er = change / volatility
    er = er.fillna(0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to 12h timeframe (wait for 1d bar to close)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI (14-period) on 12h closes
    rsi_period = 14
    delta = pd.Series(prices['close'].values).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # fill neutral when undefined
    
    # Volume confirmation (volume spike > 2.0x 50-period average)
    vol_ma_50 = pd.Series(prices['volume'].values).rolling(window=50, min_periods=50).mean().values
    vol_ratio = prices['volume'].values / vol_ma_50
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price above KAMA + RSI crosses above 50 + volume spike
            if (price_close > kama_val and 
                rsi_val > 50 and 
                rsi[i-1] <= 50 and  # RSI crossed above 50
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA + RSI crosses below 50 + volume spike
            elif (price_close < kama_val and 
                  rsi_val < 50 and 
                  rsi[i-1] >= 50 and  # RSI crossed below 50
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal OR RSI returns to neutral zone
            exit_signal = False
            
            # Trend reversal exit
            if position == 1 and price_close < kama_val:
                exit_signal = True
            elif position == -1 and price_close > kama_val:
                exit_signal = True
            
            # RSI mean reversion exit (return to 50)
            if position == 1 and rsi_val < 50:
                exit_signal = True
            if position == -1 and rsi_val > 50:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_KAMA_RSI_Volume_Momentum"
timeframe = "12h"
leverage = 1.0