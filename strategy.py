#!/usr/bin/env python3
# 6h_1w_vwap_reversion_v1
# Hypothesis: Mean reversion from weekly VWAP deviation on 6h chart. Price tends to revert to weekly VWAP after significant deviations, especially when combined with volume confirmation and volatility filtering. Works in both bull and bear markets as mean reversion is a universal market behavior. Uses 1-week VWAP as dynamic support/resistance.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_vwap_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(14) for volatility filter
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr = np.zeros(n)
    atr[0] = tr[0]
    for i in range(1, n):
        atr[i] = 0.9 * atr[i-1] + 0.1 * tr[i]
    
    # Load 1w data ONCE before loop for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly VWAP
    typical_price = (df_1w['high'] + df_1w['low'] + df_1w['close']) / 3
    vwap_num = (typical_price * df_1w['volume']).cumsum()
    vwap_den = df_1w['volume'].cumsum()
    vwap = vwap_num / vwap_den
    
    # Align weekly VWAP to 6h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1w, vwap.values)
    
    # Volume confirmation - 24 period average (4 days worth of 6h bars)
    vol_ma_24 = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 24:
            vol_sum -= volume[i-24]
        if i >= 23:
            vol_ma_24[i] = vol_sum / 24
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(atr[i]) or np.isnan(vwap_aligned[i]) or np.isnan(vol_ma_24[i]):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility
        vol_filter = atr[i] < 0.06 * close[i]  # ATR less than 6% of price
        
        # Volume confirmation: current volume > 1.2x 24-period average
        vol_ok = volume[i] > vol_ma_24[i] * 1.2
        
        # Calculate deviation from weekly VWAP as percentage
        vwap_dev_pct = (close[i] - vwap_aligned[i]) / vwap_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price returns to VWAP (mean reversion complete) or adverse move
            if vwap_dev_pct > 0.005:  # Price above VWAP by 0.5%
                position = 0
                signals[i] = 0.0
            elif close[i] < close[i-1] and vwap_dev_pct < -0.02:  # Stop loss if downtrend continues
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to VWAP or adverse move
            if vwap_dev_pct < -0.005:  # Price below VWAP by 0.5%
                position = 0
                signals[i] = 0.0
            elif close[i] > close[i-1] and vwap_dev_pct > 0.02:  # Stop loss if uptrend continues
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: price significantly below VWAP with volume confirmation
            if vwap_dev_pct < -0.015 and vol_ok and vol_filter:  # 1.5% below VWAP
                position = 1
                signals[i] = 0.25
            # Enter short: price significantly above VWAP with volume confirmation
            elif vwap_dev_pct > 0.015 and vol_ok and vol_filter:  # 1.5% above VWAP
                position = -1
                signals[i] = -0.25
    
    return signals