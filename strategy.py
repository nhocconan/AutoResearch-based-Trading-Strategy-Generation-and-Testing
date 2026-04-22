# 77246
#!/usr/bin/env python3
"""
Hypothesis: 12-hour 50-period moving average crossover with 1-day volatility filter and volume confirmation.
Long when 12h EMA50 crosses above SMA200 and 1-day ATR ratio > 1.5 (high volatility) with volume > 1.3x average.
Short when 12h EMA50 crosses below SMA200 and 1-day ATR ratio > 1.5 with volume > 1.3x average.
Exit when EMA50 crosses back in opposite direction or volatility drops.
Designed for low frequency (~20-30 trades/year) with volatility filter to capture trending moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMAs and SMAs on 12h timeframe
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    sma200 = pd.Series(close).rolling(window=200, min_periods=200).mean().values
    
    # Load 1-day data for volatility filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 14-day ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate ATR ratio (current ATR / 50-period average ATR) to detect volatility expansion
    atr_ma50 = pd.Series(atr14).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma50 > 0, atr14 / atr_ma50, 1.0)
    
    # Align volatility ratio to 12h timeframe
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Volume filter: 20-period average volume on 12h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50[i]) or np.isnan(sma200[i]) or 
            np.isnan(atr_ratio_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: EMA50 crosses above SMA200 with high volatility and volume confirmation
            if (ema50[i] > sma200[i] and ema50[i-1] <= sma200[i-1] and 
                atr_ratio_aligned[i] > 1.5 and volume[i] > 1.3 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: EMA50 crosses below SMA200 with high volatility and volume confirmation
            elif (ema50[i] < sma200[i] and ema50[i-1] >= sma200[i-1] and 
                  atr_ratio_aligned[i] > 1.5 and volume[i] > 1.3 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: EMA50 crosses below SMA200 OR volatility contraction
                if (ema50[i] < sma200[i] and ema50[i-1] >= sma200[i-1]) or atr_ratio_aligned[i] < 1.2:
                    exit_signal = True
            else:  # position == -1
                # Exit short: EMA50 crosses above SMA200 OR volatility contraction
                if (ema50[i] > sma200[i] and ema50[i-1] <= sma200[i-1]) or atr_ratio_aligned[i] < 1.2:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_EMA50_SMA200_Volatility_VolumeFilter"
timeframe = "12h"
leverage = 1.0