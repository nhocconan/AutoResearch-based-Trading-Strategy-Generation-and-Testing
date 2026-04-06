#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d weekly breakout with volume and volatility filter
# Enter long when: price breaks above 20-day high with volume > 1.8x average and low volatility (ATR ratio < 0.8)
# Enter short when: price breaks below 20-day low with volume > 1.8x average and low volatility (ATR ratio < 0.8)
# Exit when: price reverses 50% of the breakout range or volatility expands (ATR ratio > 1.2)
# Targets 20-40 trades over 4 years on 1d timeframe to minimize fee drag and capture strong trends

name = "1d_weekly_breakout_vol_volatility_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 20-period high/low for breakout levels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    # ATR(14) for volatility measurement and stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR ratio: current ATR / 50-period average ATR (volatility regime filter)
    atr_ma = pd.Series(atr).rolling(window=50, min_periods=50).mean().values
    atr_ratio = np.where(atr_ma > 0, atr / atr_ma, 1.0)
    
    # Weekly trend filter (1w close > 1w SMA 50 for uptrend, < for downtrend)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    sma_50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    sma_50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    breakout_level = 0.0  # Track breakout level for exit
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(atr_ratio[i]) or
            np.isnan(sma_50_1w_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price retrace 50% of breakout range OR volatility expands
            if close[i] <= breakout_level or atr_ratio[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price retrace 50% of breakout range OR volatility expands
            if close[i] >= breakout_level or atr_ratio[i] > 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price breaks 20-day high/low + volume + low volatility + weekly trend alignment
            if volume[i] > volume_threshold[i] and atr_ratio[i] < 0.8:
                if close[i] > high_20[i] and close[i] > sma_50_1w_aligned[i]:
                    # Bullish breakout above 20-day high with weekly uptrend
                    signals[i] = 0.25
                    position = 1
                    breakout_level = high_20[i]  # Store breakout level for exit
                elif close[i] < low_20[i] and close[i] < sma_50_1w_aligned[i]:
                    # Bearish breakout below 20-day low with weekly downtrend
                    signals[i] = -0.25
                    position = -1
                    breakout_level = low_20[i]  # Store breakdown level for exit
    
    return signals