#!/usr/bin/env python3
name = "6h_1d_200EMA_Pullback_Momentum"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Daily EMA(200) for long-term trend filter
    ema_200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 6h RSI(14) for momentum
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 6h EMA(20) for short-term trend
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20, 14)  # Wait for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_200_1d_aligned[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above daily EMA200, RSI > 50, price above EMA20, volume confirmation
            if (close[i] > ema_200_1d_aligned[i] and 
                rsi[i] > 50 and 
                close[i] > ema_20[i] and 
                volume[i] > vol_ma_20[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price below daily EMA200, RSI < 50, price below EMA20, volume confirmation
            elif (close[i] < ema_200_1d_aligned[i] and 
                  rsi[i] < 50 and 
                  close[i] < ema_20[i] and 
                  volume[i] > vol_ma_20[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below EMA20 or RSI drops below 40
            if close[i] < ema_20[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above EMA20 or RSI rises above 60
            if close[i] > ema_20[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s EMA20 pullback strategy with daily EMA200 trend filter
# - Uses daily EMA200 to determine long-term trend (bull/bear filter)
# - Enters long when price is above daily EMA200, RSI > 50, and pulls back to EMA20 with volume
# - Enters short when price is below daily EMA200, RSI < 50, and bounces to EMA20 with volume
# - Works in both bull and bear markets by aligning with higher timeframe trend
# - Volume confirmation (1.5x average) ensures institutional participation
# - Exits on EMA20 crossover or RSI exhaustion to avoid overstaying
# - Position size 0.25 targets ~20-50 trades/year, minimizing fee drag
# - Designed for BTC/ETH which respect EMA200 as major support/resistance
# - Uses momentum (RSI) to avoid buying into strong downtrends or selling into strong uptrends
# - Simple, robust logic with minimal overfitting risk
# - Aims for 50-150 total trades over 4 years (12-37/year) on BTC/ETH/SOL