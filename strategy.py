#!/usr/bin/env python3
# 4h_1d_12h_RSI_Divergence_Momentum
# Hypothesis: Combines 1d RSI divergence with 12h momentum for high-probability reversals in both bull and bear markets.
# Uses 1d RSI divergence (price makes new high/low but RSI does not) as early reversal signal.
# Confirms with 12h momentum (ROC > 0 for long, < 0 for short) and volume spike.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by catching reversals at extremes.

name = "4h_1d_12h_RSI_Divergence_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike: >1.5x 20-period average (on 4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # Daily data for RSI divergence
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 14-period RSI
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Calculate RSI divergence: bearish (price HH, RSI LH) and bullish (price LL, RSI HL)
    lookback = 10  # bars to look back for swing points
    bearish_div = np.full(len(rsi_values), False)
    bullish_div = np.full(len(rsi_values), False)
    
    for i in range(lookback, len(rsi_values)):
        # Bearish divergence: price makes higher high, RSI makes lower high
        if (high_1d[i] == np.max(high_1d[i-lookback:i+1]) and 
            rsi_values[i] < np.max(rsi_values[i-lookback:i])):
            bearish_div[i] = True
        # Bullish divergence: price makes lower low, RSI makes higher low
        if (low_1d[i] == np.min(low_1d[i-lookback:i+1]) and 
            rsi_values[i] > np.min(rsi_values[i-lookback:i])):
            bullish_div[i] = True
    
    # 12h data for momentum (ROC)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # 10-period ROC
    roc = np.full(len(close_12h), np.nan)
    for i in range(10, len(close_12h)):
        roc[i] = (close_12h[i] - close_12h[i-10]) / close_12h[i-10] * 100
    
    # Align all indicators to 4h timeframe
    bearish_div_aligned = align_htf_to_ltf(prices, df_1d, bearish_div)
    bullish_div_aligned = align_htf_to_ltf(prices, df_1d, bullish_div)
    roc_aligned = align_htf_to_ltf(prices, df_12h, roc)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if (np.isnan(bearish_div_aligned[i]) or
            np.isnan(bullish_div_aligned[i]) or
            np.isnan(roc_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Bullish RSI divergence + positive 12h ROC + volume spike
            if (bullish_div_aligned[i] and 
                roc_aligned[i] > 0 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Bearish RSI divergence + negative 12h ROC + volume spike
            elif (bearish_div_aligned[i] and 
                  roc_aligned[i] < 0 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Bearish RSI divergence OR 12h ROC turns negative
            if bearish_div_aligned[i] or roc_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Bullish RSI divergence OR 12h ROC turns positive
            if bullish_div_aligned[i] or roc_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals