#!/usr/bin/env python3
# 6h_KAMA_Trend_Filter_With_RSI_Pullback
# Hypothesis: Uses KAMA trend direction from 1d timeframe to filter entries on 6h timeframe.
# Enters on RSI pullback (30 for long, 70 for short) in the direction of the higher timeframe trend.
# Includes volume confirmation to avoid false signals. Designed to work in both bull and bear markets
# by following the higher timeframe trend and entering on pullbacks.
# Position size 0.25 for balanced risk. Targets 15-30 trades per year to avoid fee drag.

name = "6h_KAMA_Trend_Filter_With_RSI_Pullback"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA trend filter and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d timeframe
    # Efficiency Ratio (ER)
    change = abs(df_1d['close'].diff(10).values)
    volatility = df_1d['close'].diff().abs().rolling(window=10).sum().values
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(df_1d['close'].values)
    kama[0] = df_1d['close'].iloc[0]
    for i in range(1, len(kama)):
        kama[i] = kama[i-1] + sc[i] * (df_1d['close'].iloc[i] - kama[i-1])
    
    # Align KAMA to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # Calculate RSI on 6h timeframe
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Warmup for volume MA and RSI
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d KAMA
        uptrend = close[i] > kama_aligned[i]
        downtrend = close[i] < kama_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: RSI pullback (30) in uptrend with volume confirmation
            if rsi[i] < 30 and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: RSI pullback (70) in downtrend with volume confirmation
            elif rsi[i] > 70 and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: RSI reaches overbought (70) or trend turns down
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: RSI reaches oversold (30) or trend turns up
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals