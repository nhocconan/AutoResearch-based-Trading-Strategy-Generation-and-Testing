#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w KAMA trend filter with 1d RSI mean reversion and volume confirmation.
In uptrend (price > weekly KAMA), buy when RSI < 30 and volume > 1.5x average; in downtrend (price < weekly KAMA), 
sell when RSI > 70 and volume > 1.5x average. Weekly KAMA filters noise and adapts to volatility.
1d RSI identifies overextended moves; volume confirms reversal strength. Designed for 1d to target 30-100 total trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def kama(close, er_fast=2, er_slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.subtract.accumulate(change))
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(er_fast+1) - 2/(er_slow+1)) + 2/(er_slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for KAMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 1w KAMA for trend filter (adaptive to volatility)
    close_1w = df_1w['close'].values
    kama_1w = kama(close_1w, er_fast=2, er_slow=30)
    kama_1w_aligned = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # 1d RSI for mean reversion
    close_1d = prices['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume confirmation (volume spike > 1.5x 20-period average)
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        kama_trend = kama_1w_aligned[i]
        rsi_val = rsi[i]
        vol_ratio_val = vol_ratio[i]
        vol_threshold = 1.5  # Volume spike filter for quality
        
        if position == 0:
            # Enter long: price above weekly KAMA (uptrend) + RSI oversold + volume spike
            if (price_close > kama_trend and 
                rsi_val < 30 and 
                vol_ratio_val > vol_threshold):
                signals[i] = 0.25
                position = 1
            # Enter short: price below weekly KAMA (downtrend) + RSI overbought + volume spike
            elif (price_close < kama_trend and 
                  rsi_val > 70 and 
                  vol_ratio_val > vol_threshold):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: trend reversal (price crosses weekly KAMA in opposite direction) or RSI normalization
            if position == 1 and (price_close < kama_trend or rsi_val > 70):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > kama_trend or rsi_val < 30):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_KAMA_RSI_MeanReversion_Volume"
timeframe = "1d"
leverage = 1.0