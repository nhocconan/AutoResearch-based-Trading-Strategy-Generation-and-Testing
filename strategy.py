#!/usr/bin/env python3
"""
6h_1w_1d_Momentum_Reversal_With_Volume
Hypothesis: Combines weekly momentum (RSI) with daily mean reversion (CCI) on 6h timeframe.
- Long when: Weekly RSI < 40 (weak momentum) AND Daily CCI < -100 (oversold) AND 6h volume spike
- Short when: Weekly RSI > 60 (strong momentum) AND Daily CCI > 100 (overbought) AND 6h volume spike
- Uses volume spike to confirm momentum shift
- Designed for low trade frequency (20-50 trades/year) to work in both bull and bear markets
- Weekly RSI avoids chasing momentum; Daily CCI captures mean reversion opportunities
"""

name = "6h_1w_1d_Momentum_Reversal_With_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume spike: >2x 50-period average (on 6h timeframe)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Weekly data for RSI momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Calculate RSI(14) on weekly data
    delta = pd.Series(close_1w).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w_values = rsi_1w.values
    
    # Daily data for CCI mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate CCI(20) on daily data
    tp_1d = (high_1d + low_1d + close_1d) / 3.0
    sma_tp = pd.Series(tp_1d).rolling(window=20, min_periods=20).mean()
    mad = pd.Series(tp_1d).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci_1d = (tp_1d - sma_tp) / (0.015 * mad)
    cci_1d_values = cci_1d.values
    
    # Align all indicators to 6h timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w_values)
    cci_1d_aligned = align_htf_to_ltf(prices, df_1d, cci_1d_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(cci_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Weak weekly momentum + daily oversold + volume spike
            if (rsi_1w_aligned[i] < 40 and 
                cci_1d_aligned[i] < -100 and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Strong weekly momentum + daily overbought + volume spike
            elif (rsi_1w_aligned[i] > 60 and 
                  cci_1d_aligned[i] > 100 and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Weekly momentum strengthens OR daily overbought
            if (rsi_1w_aligned[i] > 50) or \
               (cci_1d_aligned[i] > 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Weekly momentum weakens OR daily oversold
            if (rsi_1w_aligned[i] < 50) or \
               (cci_1d_aligned[i] < 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals