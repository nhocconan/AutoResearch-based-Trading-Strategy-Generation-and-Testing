#!/usr/bin/env python3
# Hypothesis: 6h price action near 1d volume-weighted average price (VWAP) with volume confirmation and trend filter
# Long when price is above daily VWAP, above 6h EMA34, and volume > 1.8x 20-period average
# Short when price is below daily VWAP, below 6h EMA34, and volume > 1.8x 20-period average
# Exit when price crosses back below/above VWAP OR EMA direction contradicts position
# Position size: 0.25 (25% of capital) to balance return and drawdown
# Designed to work in trending markets via EMA filter and in ranging markets via VWAP reversion

name = "6h_VWAP_EMA_Volume_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h EMA34 for trend filter
    ema34 = pd.Series(close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 1d data for VWAP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily VWAP: sum(price * volume) / sum(volume) for the day
    # Using typical price (H+L+C)/3 * volume
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_numerator = (typical_price * df_1d['volume']).cumsum()
    vwap_denominator = df_1d['volume'].cumsum()
    vwap = vwap_numerator / vwap_denominator
    
    # Align 1d VWAP to 6h timeframe (waits for daily close)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap.values)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.8 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough data for EMA34
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34[i]) or np.isnan(vwap_aligned[i]) or 
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above VWAP AND above EMA34 (bullish alignment) + volume spike
            if (close[i] > vwap_aligned[i] and 
                close[i] > ema34[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below VWAP AND below EMA34 (bearish alignment) + volume spike
            elif (close[i] < vwap_aligned[i] and 
                  close[i] < ema34[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below VWAP OR EMA34 turns bearish
            if (close[i] < vwap_aligned[i]) or (close[i] < ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above VWAP OR EMA34 turns bullish
            if (close[i] > vwap_aligned[i]) or (close[i] > ema34[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals