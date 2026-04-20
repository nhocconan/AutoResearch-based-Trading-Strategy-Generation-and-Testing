#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h with 1d RSI(14) and 1w Chaikin Oscillator for mean reversion in range-bound markets.
# Long when RSI < 30 and Chaikin Oscillator > 0 (accumulation). Short when RSI > 70 and Chaikin < 0 (distribution).
# Exit when RSI returns to neutral (40-60). Uses volume accumulation/distribution to filter false signals.
# Target: 30-60 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    alpha = 1.0 / 14
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[0] = gain[0]
    avg_loss[0] = loss[0]
    for i in range(1, len(gain)):
        avg_gain[i] = alpha * gain[i] + (1 - alpha) * avg_gain[i-1]
        avg_loss[i] = alpha * loss[i] + (1 - alpha) * avg_loss[i-1]
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[0] = 50  # neutral for first value
    
    # Load 1w data for Chaikin Oscillator
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Money Flow Multiplier
    mfm = ((close_1w - low_1w) - (high_1w - close_1w)) / np.where((high_1w - low_1w) == 0, 1, (high_1w - low_1w))
    mfv = mfm * volume_1w
    
    # Chaikin Oscillator = (3-period EMA of MFV) - (10-period EMA of MFV)
    ema3 = pd.Series(mfv).ewm(span=3, adjust=False, min_periods=3).mean().values
    ema10 = pd.Series(mfv).ewm(span=10, adjust=False, min_periods=10).mean().values
    chaikin = ema3 - ema10
    
    # Align indicators to 6h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    chaikin_aligned = align_htf_to_ltf(prices, df_1w, chaikin)
    
    # 6h data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if np.isnan(rsi_aligned[i]) or np.isnan(chaikin_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        rsi_val = rsi_aligned[i]
        chaikin_val = chaikin_aligned[i]
        
        if position == 0:
            # Long: oversold RSI + accumulation
            if rsi_val < 30 and chaikin_val > 0:
                signals[i] = 0.25
                position = 1
            # Short: overbought RSI + distribution
            elif rsi_val > 70 and chaikin_val < 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI returns to neutral (40-60)
            if 40 <= rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI returns to neutral (40-60)
            if 40 <= rsi_val <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_RSI14_1w_ChaikinOscillator_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0