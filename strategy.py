#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d RSI filter and volume confirmation
# Uses 1-day RSI to filter for momentum exhaustion (RSI<30 for long, RSI>70 for short)
# Donchian(20) breakout captures momentum, volume confirms strength
# Target: 15-25 trades/year per symbol with disciplined entries
# Works in both bull (breakouts continue) and bear (mean reversion at extremes)
name = "12h_Donchian20_1dRSI_Volume"
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
    
    # 1d RSI for momentum filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate RSI on daily closes
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Donchian channel (20-period) on 12h data
    lookback = 20
    upper = np.full_like(high, np.nan, dtype=float)
    lower = np.full_like(low, np.nan, dtype=float)
    
    for i in range(lookback - 1, len(high)):
        upper[i] = np.max(high[i-lookback+1:i+1])
        lower[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Need enough data for Donchian and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian band + RSI not overbought + volume confirmation
            if (close[i] > upper[i] and 
                rsi_1d_aligned[i] < 70 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band + RSI not oversold + volume confirmation
            elif (close[i] < lower[i] and 
                  rsi_1d_aligned[i] > 30 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian band or RSI becomes overbought
            if (close[i] < lower[i]) or (rsi_1d_aligned[i] >= 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian band or RSI becomes oversold
            if (close[i] > upper[i]) or (rsi_1d_aligned[i] <= 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals