#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_kama_rsi_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for higher timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    close_1d = df_1d['close'].values
    close_1d_series = pd.Series(close_1d)
    # Efficiency Ratio
    change = abs(close_1d_series - close_1d_series.shift(10))
    volatility = abs(close_1d_series.diff()).rolling(window=10, min_periods=10).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    # KAMA calculation
    kama_1d = np.full_like(close_1d, np.nan, dtype=np.float64)
    kama_1d[9] = close_1d[9]  # Initialize
    for i in range(10, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc.iloc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Align KAMA to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI on 1d close
    delta = close_1d_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d.values)
    
    # Calculate Choppy Index on 1d for regime detection
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift())
    tr3 = abs(df_1d['low'] - df_1d['close'].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr.rolling(window=14, min_periods=14).sum()
    highest_high = df_1d['high'].rolling(window=14, min_periods=14).max()
    lowest_low = df_1d['low'].rolling(window=14, min_periods=14).min()
    chop_1d = 100 * np.log10(atr_1d / (highest_high - lowest_low)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d.values)
    
    # Volume filter on 12h
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(volume_ok[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price above KAMA, RSI > 50, choppy market (chop > 61.8)
        long_signal = (close[i] > kama_1d_aligned[i] and 
                      rsi_1d_aligned[i] > 50 and 
                      chop_1d_aligned[i] > 61.8 and 
                      volume_ok[i])
        
        # Short: price below KAMA, RSI < 50, choppy market (chop > 61.8)
        short_signal = (close[i] < kama_1d_aligned[i] and 
                       rsi_1d_aligned[i] < 50 and 
                       chop_1d_aligned[i] > 61.8 and 
                       volume_ok[i])
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif (position == 1 and close[i] < kama_1d_aligned[i]) or \
             (position == -1 and close[i] > kama_1d_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals