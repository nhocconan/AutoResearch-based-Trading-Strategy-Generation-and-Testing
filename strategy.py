#!/usr/bin/env python3
"""
4h_RSI_Stochastic_Divergence_1dTrend
Hypothesis: RSI(14) divergence combined with Stochastic(14,3,3) oversold/overbought levels and 1-day trend filter.
In uptrend (price > 1d EMA50): long on bullish RSI divergence + Stochastic oversold.
In downtrend (price < 1d EMA50): short on bearish RSI divergence + Stochastic overbought.
Uses volume confirmation to filter weak signals. Target: 20-30 trades per year (~80-120 over 4 years) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_RSI_Stochastic_Divergence_1dTrend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Stochastic(14,3,3)
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = np.where(highest_high - lowest_low != 0, 
                         (close - lowest_low) / (highest_high - lowest_low) * 100, 50)
    d_percent = pd.Series(k_percent).rolling(window=3, min_periods=3).mean().values
    
    # Volume confirmation: volume > 1.3x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Need 30 periods for RSI and Stochastic
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(rsi[i]) or np.isnan(d_percent[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine market trend from 1-day EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation
        volume_confirm = vol_ratio[i] > 1.3
        
        # RSI divergence detection (bullish/bearish)
        bullish_div = False
        bearish_div = False
        
        if i >= 5:  # Need at least 5 periods to detect divergence
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-3] and low[i-3] < low[i-6] and 
                rsi[i] > rsi[i-3] and rsi[i-3] > rsi[i-6]):
                bullish_div = True
            # Bearish divergence: price makes higher high, RSI makes lower high
            if (high[i] > high[i-3] and high[i-3] > high[i-6] and 
                rsi[i] < rsi[i-3] and rsi[i-3] < rsi[i-6]):
                bearish_div = True
        
        if position == 0:
            # Long: bullish RSI divergence + Stochastic oversold (<30) in uptrend + volume
            long_entry = bullish_div and (d_percent[i] < 30) and uptrend and volume_confirm
            # Short: bearish RSI divergence + Stochastic overbought (>70) in downtrend + volume
            short_entry = bearish_div and (d_percent[i] > 70) and downtrend and volume_confirm
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI crosses above 70 (overbought) or trend changes to downtrend
            if (rsi[i] > 70) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI crosses below 30 (oversold) or trend changes to uptrend
            if (rsi[i] < 30) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals