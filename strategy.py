#!/usr/bin/env python3
name = "4h_KAMA_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    close_1d = df_1d['close'].values
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    er = np.nan_to_num(er, nan=0.0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama = np.nan_to_num(kama, nan=close_1d[0])
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 4h Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high, above daily KAMA, volume spike
            if (close[i] > high_roll[i] and 
                close[i] > kama_aligned[i] and 
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below daily KAMA, volume spike
            elif (close[i] < low_roll[i] and 
                  close[i] < kama_aligned[i] and 
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low or below daily KAMA
            if (close[i] < low_roll[i] or 
                close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high or above daily KAMA
            if (close[i] > high_roll[i] or 
                close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA breakout with volume confirmation.
# KAMA adapts to market noise - slows in ranging markets, speeds in trends.
# Breakout above 4h Donchian high with volume and price > daily KAMA signals bullish momentum.
# Breakdown below 4h Donchian low with volume and price < daily KAMA signals bearish momentum.
# Works in bull markets (buy breakouts above Donchian high in uptrend) and bear markets 
# (sell breakdowns below Donchian low in downtrend).
# Position size 0.25 balances risk and keeps trade frequency manageable (~15-30 trades/year).