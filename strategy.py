#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour momentum strategy using 1-day RSI (14) for regime detection and 
# 4-hour Donchian (20) breakout for entry. In bull markets (RSI > 50), we look for long 
# breakouts; in bear markets (RSI < 50), we look for short breakouts. Volume > 1.5x 
# 20-period average confirms institutional participation. This adapts to both bull and 
# bear markets by using the 1-day RSI to determine market bias, reducing counter-trend 
# trades. Target: 20-40 trades/year per symbol (80-160 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for RSI regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d RSI(14) for regime detection
    rsi_len = 14
    if len(df_1d) < rsi_len:
        return np.zeros(n)
    
    close_1d = pd.Series(df_1d['close'])
    delta = close_1d.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean()
    avg_loss = loss.ewm(alpha=1/rsi_len, adjust=False, min_periods=rsi_len).mean()
    rs = avg_gain / avg_loss
    rsi_1d = (100 - (100 / (1 + rs))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Donchian channel (20 periods) on 4h
    dc_len = 20
    dc_upper = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().shift(1).values
    dc_lower = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().shift(1).values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20, rsi_len)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_upper[i]) or 
            np.isnan(dc_lower[i]) or
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 1d RSI > 50 = bull bias, < 50 = bear bias
        bull_bias = rsi_1d_aligned[i] > 50
        bear_bias = rsi_1d_aligned[i] < 50
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: Donchian breakout above + bull bias + volume
            if (close[i] > dc_upper[i] and 
                bull_bias and 
                volume_confirmed):
                position = 1
                signals[i] = position_size
            # Enter short: Donchian breakdown below + bear bias + volume
            elif (close[i] < dc_lower[i] and 
                  bear_bias and 
                  volume_confirmed):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to Donchian lower or breaks below
            if close[i] < dc_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to Donchian upper or breaks above
            if close[i] > dc_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI_Donchian_Volume_v1"
timeframe = "4h"
leverage = 1.0