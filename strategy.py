#!/usr/bin/env python3
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Williams %R (14)
    highest_high = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close']) / (highest_high - lowest_low + 1e-10)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 1d RSI(14)
    delta = np.diff(df_1d['close'], prepend=df_1d['close'].iloc[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(williams_r_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Williams %R: oversold < -80, overbought > -20
        wr = williams_r_aligned[i]
        rsi = rsi_1d_aligned[i]
        
        # Mean reversion conditions
        oversold = wr < -80
        overbought = wr > -20
        
        if position == 0:
            # Long setup: oversold + RSI not extremely low (avoid catching falling knife)
            if oversold and rsi > 30:
                position = 1
                signals[i] = position_size
            # Short setup: overbought + RSI not extremely high
            elif overbought and rsi < 70:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: overbought OR RSI > 75 (strong momentum)
            if overbought or rsi > 75:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: oversold OR RSI < 25 (strong momentum)
            if oversold or rsi < 25:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1dWilliamsR_RSI_MeanReversion_v1"
timeframe = "12h"
leverage = 1.0