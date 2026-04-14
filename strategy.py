#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h EMA crossover with 1d RSI filter and volume confirmation
# EMA(9/21) crossover captures momentum shifts
# 1d RSI(14) filters trades: only long when RSI<70, short when RSI>30 (avoid overextended moves)
# Volume > 1.5x average confirms institutional participation
# Works in bull/bear as EMA adapts to trend and RSI prevents chasing extremes
# Target: 12-25 trades/year per symbol (48-100 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for RSI filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d RSI(14)
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_values = rsi_1d.values
    
    # Align 1d RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d_values)
    
    # EMA(9) and EMA(21) on 6h
    ema_fast = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_slow = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 21, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_fast[i]) or 
            np.isnan(ema_slow[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # EMA crossover signals
        ema_cross_up = ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1]
        ema_cross_down = ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1]
        
        # RSI filters: avoid overbought/oversold extremes
        rsi_not_overbought = rsi_1d_aligned[i] < 70
        rsi_not_oversold = rsi_1d_aligned[i] > 30
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: EMA bullish crossover + RSI not overbought + volume
            if ema_cross_up and rsi_not_overbought and volume_confirmed:
                position = 1
                signals[i] = position_size
            # Enter short: EMA bearish crossover + RSI not oversold + volume
            elif ema_cross_down and rsi_not_oversold and volume_confirmed:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: EMA bearish crossover or RSI overbought
            if ema_cross_down or rsi_1d_aligned[i] >= 70:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: EMA bullish crossover or RSI oversold
            if ema_cross_up or rsi_1d_aligned[i] <= 30:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_EMA_RSI_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0