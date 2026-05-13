#!/usr/bin/env python3
"""
4h_RSI_Pullback_Trend_With_Volume_Filter
Hypothesis: In strong trends, RSI pullbacks offer high-probability entries. Uses RSI(14) < 30 for longs and > 70 for shorts in the direction of the 1D EMA50 trend, with volume confirmation (>1.5x 20-bar average) to filter low-quality signals. Designed for low trade frequency (~20-30/year) to avoid fee drag, with discrete position sizing (0.25) to manage drawdown. Works in bull/bear by following the higher timeframe trend.
"""

name = "4h_RSI_Pullback_Trend_With_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1D data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate RSI(14)
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral before warmup
    
    # 1D trend filter: EMA(50) on close
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start after RSI warmup
        if position == 0:
            # LONG: RSI < 30 (oversold), volume confirmation, price above 1D EMA50 (uptrend)
            if (rsi_values[i] < 30 and 
                volume_filter[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (overbought), volume confirmation, price below 1D EMA50 (downtrend)
            elif (rsi_values[i] > 70 and 
                  volume_filter[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI > 70 (overbought) OR volume drops
            if (rsi_values[i] > 70) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 30 (oversold) OR volume drops
            if (rsi_values[i] < 30) or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals