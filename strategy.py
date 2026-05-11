# 1d_TurtleTrader_System_v1
# Hypothesis: Turtle Trading system adapted for daily timeframe with Donchian breakouts, ATR-based position sizing, and volatility filtering. Works in trending markets (both bull and bear) by catching breakouts with strict risk management. Uses 20-day Donchian for entry and 10-day for exit, with ATR stops. Target: 15-25 trades per year on 1d timeframe.

name = "1d_TurtleTrader_System_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 1D Data for Turtle System ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channels: 20-day for entry, 10-day for exit
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_10 = pd.Series(high_1d).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low_1d).rolling(window=10, min_periods=10).min().values
    
    # ATR for volatility filtering and position sizing
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First day
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 1d timeframe (already aligned, but for consistency)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    high_10_aligned = align_htf_to_ltf(prices, df_1d, high_10)
    low_10_aligned = align_htf_to_ltf(prices, df_1d, low_10)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or 
            np.isnan(high_10_aligned[i]) or 
            np.isnan(low_10_aligned[i]) or 
            np.isnan(atr_aligned[i]) or
            atr_aligned[i] <= 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high
            if close[i] > high_20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low
            elif close[i] < low_20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below 10-day low OR ATR-based stop
            if close[i] < low_10_aligned[i] or close[i] < (high[i] - 2.0 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price breaks above 10-day high OR ATR-based stop
            if close[i] > high_10_aligned[i] or close[i] > (low[i] + 2.0 * atr_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals