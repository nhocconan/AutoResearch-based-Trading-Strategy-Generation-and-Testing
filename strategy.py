#!/usr/bin/env python3
"""
4h_1d_RSI_Overbought_Oversold
Hypothesis: Mean reversion from RSI extremes (overbought >70, oversold <30) on 4h timeframe, filtered by 1d trend (price above/below EMA200) and volume confirmation. Works in both bull and bear markets by capturing overextended moves that revert to the mean. RSI extremes often occur at trend exhaustion points, providing high-probability reversals. Volume ensures genuine participation, not just low-liquidity spikes. Target: 20-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate RSI(14) on 4h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Get daily data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(200, n):
        # Skip if any required data is not ready
        if (np.isnan(rsi[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Long: RSI oversold (<30) with volume expansion and price above daily EMA200
        long_condition = (rsi[i] < 30) and volume_expansion[i] and (close[i] > ema_200_aligned[i])
        
        # Short: RSI overbought (>70) with volume expansion and price below daily EMA200
        short_condition = (rsi[i] > 70) and volume_expansion[i] and (close[i] < ema_200_aligned[i])
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_1d_RSI_Overbought_Oversold"
timeframe = "4h"
leverage = 1.0