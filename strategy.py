#!/usr/bin/env python3
# Hypothesis: 12h price action relative to daily VWAP with volume confirmation and 1-day trend filter
# Long when price > daily VWAP, price > 1-day EMA200, and volume > 2x 20-period average
# Short when price < daily VWAP, price < 1-day EMA200, and volume > 2x 20-period average
# Exit when price crosses VWAP or EMA200 trend contradicts position
# Position size: 0.25 to limit drawdown and reduce trade frequency
# Designed to work in trending markets via EMA200 filter and in ranging markets via VWAP mean reversion

name = "12h_VWAP_EMA200_Volume_Filter"
timeframe = "12h"
leverage = 1.0

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
    
    # 12-period VWAP approximation for 12h timeframe (using typical price)
    typical_price = (high + low + close) / 3
    vwap_num = np.cumsum(typical_price * volume)
    vwap_den = np.cumsum(volume)
    vwap = vwap_num / vwap_den
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1-day EMA200
    ema200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1-day EMA200 to 12h timeframe (waits for daily close)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Need enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vwap[i]) or np.isnan(ema200_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above VWAP AND above EMA200 (bullish alignment) + volume spike
            if (close[i] > vwap[i] and 
                close[i] > ema200_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below VWAP AND below EMA200 (bearish alignment) + volume spike
            elif (close[i] < vwap[i] and 
                  close[i] < ema200_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP OR EMA200 turns bearish
            if (close[i] < vwap[i]) or (close[i] < ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP OR EMA200 turns bullish
            if (close[i] > vwap[i]) or (close[i] > ema200_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals