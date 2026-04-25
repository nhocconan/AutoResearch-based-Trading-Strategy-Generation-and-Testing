#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Confirmation
Hypothesis: On daily timeframe, use Kaufman Adaptive Moving Average (KAMA) as trend filter.
Enter long when price > KAMA with volume confirmation (>1.5x 20-day avg volume).
Enter short when price < KAMA with volume confirmation.
Exit when price crosses back over KAMA.
Volume confirmation reduces false signals in choppy markets.
Designed for low trade frequency (<25/year) to minimize fee drag and work in both bull/bear markets via trend following.
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
    
    # Get 1w data for HTF trend filter (stronger trend confirmation)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on 1d close (using 10-period ER, 2 and 30 for fast/slow SC)
    close_s = pd.Series(close)
    change = abs(close_s - close_s.shift(10)).values  # 10-period net change
    volatility = abs(close_s.diff()).rolling(window=10, min_periods=10).sum().values  # 10-period volatility
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # smoothing constant
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate EMA50 on 1w close for HTF trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1d timeframe
    kama_aligned = align_htf_to_ltf(prices, None, kama)  # KAMA is already on 1d, no alignment needed
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: 1.5x 20-day average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10) and EMA50 (50) and volume MA (20)
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Look for entry signals with HTF trend filter and volume confirmation
            # Long: price > KAMA in uptrend (close > EMA50_1w) with volume confirmation
            # Short: price < KAMA in downtrend (close < EMA50_1w) with volume confirmation
            long_signal = (close[i] > kama_aligned[i]) and (close[i] > ema50_1w_aligned[i]) and volume_spike[i]
            short_signal = (close[i] < kama_aligned[i]) and (close[i] < ema50_1w_aligned[i]) and volume_spike[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price crosses back below KAMA
            exit_signal = close[i] < kama_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price crosses back above KAMA
            exit_signal = close[i] > kama_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_Volume_Confirmation"
timeframe = "1d"
leverage = 1.0