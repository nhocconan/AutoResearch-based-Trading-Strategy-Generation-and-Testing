#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1-day ATR-based volatility regime filter
# Long when price > 200-bar EMA and price > previous high + 0.5*ATR(14) (breakout)
# Short when price < 200-bar EMA and price < previous low - 0.5*ATR(14) (breakdown)
# Uses 1-day ATR for volatility regime: only trade when ATR(14) > 0.8 * ATR(50) (high vol)
# Position size: 0.25
# Target: 15-25 trades per year (60-100 over 4 years) to avoid fee drag

name = "12h_1dATR_VolRegime_EMA200_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate EMA200 on 12h close
    close_series = pd.Series(close)
    ema200 = close_series.ewm(span=200, min_periods=200, adjust=False).mean().values
    
    # Get 1-day data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ATR(14) and ATR(50) on 1-day data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # First value has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) and ATR(50)
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr50 = pd.Series(tr).rolling(window=50, min_periods=50).mean().values
    
    # Volatility regime: high volatility when ATR(14) > 0.8 * ATR(50)
    vol_regime = atr14 > (0.8 * atr50)
    
    # Previous day's high and low for breakout levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align 1-day data to 12h timeframe
    vol_regime_aligned = align_htf_to_ltf(prices, df_1d, vol_regime)
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    atr14_aligned = align_htf_to_ltf(prices, df_1d, atr14)
    
    # Breakout threshold: 0.5 * ATR(14)
    breakout_threshold = 0.5 * atr14_aligned
    
    # Previous 12h close for momentum filter
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema200[i]) or np.isnan(vol_regime_aligned[i]) or 
            np.isnan(prev_high_aligned[i]) or np.isnan(prev_low_aligned[i]) or
            np.isnan(breakout_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above previous day's high + threshold with uptrend and high vol
            if (close[i] > prev_high_aligned[i] + breakout_threshold[i] and 
                close[i] > ema200[i] and vol_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below previous day's low - threshold with downtrend and high vol
            elif (close[i] < prev_low_aligned[i] - breakout_threshold[i] and 
                  close[i] < ema200[i] and vol_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below EMA200 or volatility drops
            if close[i] < ema200[i] or not vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above EMA200 or volatility drops
            if close[i] > ema200[i] or not vol_regime_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals