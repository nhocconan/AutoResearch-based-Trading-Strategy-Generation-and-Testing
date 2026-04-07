#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 12h Donchian(20) Breakout with 1d EMA100 Trend Filter and Volume Confirmation
# Hypothesis: Donchian breakouts capture volatility expansion moves aligned with daily trend.
# Volume confirmation ensures breakouts have institutional participation.
# Works in both bull and bear markets by only taking trades aligned with higher timeframe trend.
# Targets 15-25 trades/year with disciplined entries to avoid overtrading.

name = "12h_donchian20_1d_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA100 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    ema100_1d = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False).mean().values
    ema100_12h = align_htf_to_ltf(prices, df_1d, ema100_1d)
    
    # 12-period ATR(14) for volatility measurement (used in exit)
    tr1 = high - low
    tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
    tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # 20-period SMA for volume average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup for ATR and volume SMA
        # Skip if required data not available
        if (np.isnan(ema100_12h[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 1d EMA100 OR ATR-based trailing stop
            if close[i] < ema100_12h[i] or close[i] < high[max(0, i-3):i+1].max() - 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above 1d EMA100 OR ATR-based trailing stop
            if close[i] > ema100_12h[i] or close[i] > low[max(0, i-3):i+1].min() + 2.0 * atr[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price breaks above 20-period high + volume confirmation + uptrend
            if (close[i] > high[max(0, i-20):i].max() and 
                vol_confirm and 
                close[i] > ema100_12h[i]):
                position = 1
                signals[i] = 0.25
            # Short: price breaks below 20-period low + volume confirmation + downtrend
            elif (close[i] < low[max(0, i-20):i].min() and 
                  vol_confirm and 
                  close[i] < ema100_12h[i]):
                position = -1
                signals[i] = -0.25
    
    return signals