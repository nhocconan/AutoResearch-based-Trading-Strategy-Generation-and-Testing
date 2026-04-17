#!/usr/bin/env python3
"""
4h_Keltner_RSI_Trend_Follow
Hypothesis: Use Keltner Channel breakouts with RSI trend filter and volume confirmation to capture strong trends while avoiding chop. 
Keltner Channels adapt to volatility, reducing false breakouts in low-volatility environments. 
RSI > 50 for longs, < 50 for shorts ensures trend alignment. Volume filter ensures participation. 
Designed for low trade frequency (<30/year) to minimize fee drag and improve generalization across bull/bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4h ATR for Keltner Channel ===
    tr = np.maximum(high[1:] - low[1:], 
                    np.maximum(np.abs(high[1:] - close[:-1]), 
                               np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # === 4h EMA(20) for Keltner middle line ===
    ema20 = pd.Series(close).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # Keltner Channels: upper = EMA + 2*ATR, lower = EMA - 2*ATR
    keltner_upper = ema20 + 2.0 * atr
    keltner_lower = ema20 - 2.0 * atr
    
    # === Daily RSI for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI(14) calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align daily RSI to 4h
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # === Daily volume average for confirmation ===
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    signals = np.zeros(n)
    
    # Warmup: covers 20-day volume average, 20-period ATR/EMA, 14-day RSI
    warmup = 50
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Get current daily volume
        vol_1d_current = align_htf_to_ltf(prices, df_1d, volume_1d)[i]
        
        # Volume filter: current volume > 1.3x 20-day average
        vol_filter = vol_1d_current > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Entry conditions
        if position == 0:
            # Long: price > Keltner upper + RSI > 50 + volume
            if close[i] > keltner_upper[i] and rsi_1d_aligned[i] > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
                continue
            # Short: price < Keltner lower + RSI < 50 + volume
            elif close[i] < keltner_lower[i] and rsi_1d_aligned[i] < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit conditions: reverse signal
        elif position == 1:
            if close[i] < keltner_lower[i]:  # break below lower channel = exit long
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            if close[i] > keltner_upper[i]:  # break above upper channel = exit short
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Keltner_RSI_Trend_Follow"
timeframe = "4h"
leverage = 1.0