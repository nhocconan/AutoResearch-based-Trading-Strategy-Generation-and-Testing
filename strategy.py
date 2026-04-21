#!/usr/bin/env python3
"""
1d_1w_Momentum_Reversal_With_Volume_Filter
Hypothesis: Daily momentum reversal with weekly trend filter. Buy when daily RSI < 30 and price > weekly VWAP, sell when RSI > 70 and price < weekly VWAP. Uses volume confirmation (1.5x 20-day average) to filter low-quality signals. Designed to capture mean reversion in ranging markets while avoiding counter-trend trades in strong weekly trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for RSI and price
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align RSI to 1d timeframe (already aligned, but for consistency)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load weekly data for VWAP
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate typical price and VWAP
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    vwap_num = np.cumsum(typical_price_1w * volume_1w)
    vwap_den = np.cumsum(volume_1w)
    vwap_1w = vwap_num / (vwap_den + 1e-10)
    
    # Align VWAP to 1d timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(rsi_aligned[i]) or np.isnan(vwap_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > weekly VWAP + volume
            if rsi_aligned[i] < 30 and price > vwap_1w_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: RSI > 70 (overbought) and price < weekly VWAP + volume
            elif rsi_aligned[i] > 70 and price < vwap_1w_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI > 50 (momentum fading) or price < weekly VWAP
            if rsi_aligned[i] > 50 or price < vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI < 50 (momentum fading) or price > weekly VWAP
            if rsi_aligned[i] < 50 or price > vwap_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Momentum_Reversal_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0