#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h 4h Supertrend + 1d RSI pullback
# Use 4h Supertrend for trend direction (works in bull/bear via ATR-based dynamic support/resistance)
# Enter on 1h pullbacks to 4h Supertrend when 1d RSI is oversold/overbought (mean reversion within trend)
# Volume confirmation filters low-quality breaks
# Target: 15-35 trades/year (~60-140 total over 4 years) to avoid fee drag
# Works in bull: rides 4h uptrend, buys dips. Works in bear: shorts 4h downtrend, sells rallies.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data for Supertrend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = np.abs(high_4h[1:] - low_4h[:-1])
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.max([tr1[0], tr2[0], tr3[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_4h + low_4h) / 2
    upper = hl2 + (3.0 * atr)
    lower = hl2 - (3.0 * atr)
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper[0]
    direction[0] = 1
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_4h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper[i], supertrend[i-1])
    
    # Align Supertrend and direction to 1h
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # Load 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate RSI(14) on 1d
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad first value
    rsi = np.concatenate([[50], rsi])
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate price and volume arrays
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume filter: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or \
           np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        st = supertrend_aligned[i]
        direction_4h = direction_aligned[i]
        rsi_1d = rsi_aligned[i]
        has_volume = vol_filter[i]
        
        if position == 0:
            # Long: 4h uptrend, price near Supertrend support, 1d RSI oversold
            if direction_4h == 1 and price <= st * 1.005 and rsi_1d < 35 and has_volume:
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend, price near Supertrend resistance, 1d RSI overbought
            elif direction_4h == -1 and price >= st * 0.995 and rsi_1d > 65 and has_volume:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: 4h trend turns down OR price breaks above resistance
            if direction_4h == -1 or price >= st * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: 4h trend turns up OR price breaks below support
            if direction_4h == 1 or price <= st * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4hSupertrend_1dRSIPullback_Volume"
timeframe = "1h"
leverage = 1.0