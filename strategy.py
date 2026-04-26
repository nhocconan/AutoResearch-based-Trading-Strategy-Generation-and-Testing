#!/usr/bin/env python3
"""
6h_FundingRateMeanReversion_v1
Hypothesis: Funding rates on BTC/ETH perpetual futures exhibit mean reversion. Extreme positive funding (>0.03%) indicates overleveraged longs → short opportunity. Extreme negative funding (<-0.03%) indicates overleveraged shorts → long opportunity. Works in both bull and bear markets as funding extremes occur during strong directional moves regardless of trend. Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load funding rate data (8h timeframe) ONCE before loop
    # Note: funding data is stored in 8h parquet files
    df_8h = get_htf_data(prices, '8h')
    
    # Extract funding rate (assuming it's in the data - if not, we'll need to approximate)
    # Since we don't have direct funding data in prices, we'll use price action proxy:
    # Extreme price moves relative to volume often correlate with funding extremes
    # Alternative: use 8h RSI extremes as proxy for funding sentiment
    
    # Calculate 8h RSI as proxy for funding rate sentiment
    close_8h = df_8h['close'].values
    delta = np.diff(close_8h, prepend=close_8h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_8h = 100 - (100 / (1 + rs))
    rsi_8h_aligned = align_htf_to_ltf(prices, df_8h, rsi_8h)
    
    # Calculate volume spike indicator on 6h
    volume_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_ratio = volume / (volume_ma + 1e-10)
    
    # Price change over 2 periods to detect momentum exhaustion
    price_change_2 = np.abs(np.diff(close, 2, prepend=close[:2]))
    price_change_ma = pd.Series(price_change_2).ewm(span=20, adjust=False, min_periods=20).mean().values
    price_change_ratio = np.concatenate([[np.nan, np.nan], price_change_2[2:]]) / (price_change_ma + 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(rsi_8h_aligned[i]) or 
            np.isnan(volume_ratio[i]) or
            np.isnan(price_change_ratio[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Funding extreme proxies using RSI
        # RSI > 70 = overbought (longs overleveraged) → expect negative funding pressure → short
        # RSI < 30 = oversold (shorts overleveraged) → expect positive funding pressure → long
        rsi_overbought = rsi_8h_aligned[i] > 70
        rsi_oversold = rsi_8h_aligned[i] < 30
        
        # Volume confirmation - need participation for mean reversion to be valid
        volume_spike = volume_ratio[i] > 1.5
        
        # Price momentum exhaustion - look for weakening momentum
        momentum_exhaustion = price_change_ratio[i] < 0.5  # momentum slowing down
        
        # Long signal: oversold + volume spike + momentum exhaustion
        if rsi_oversold and volume_spike and momentum_exhaustion:
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short signal: overbought + volume spike + momentum exhaustion
        elif rsi_overbought and volume_spike and momentum_exhaustion:
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: RSI returns to neutral territory (40-60)
        elif position == 1 and (rsi_8h_aligned[i] >= 40 or not volume_spike):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (rsi_8h_aligned[i] <= 60 or not volume_spike):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_FundingRateMeanReversion_v1"
timeframe = "6h"
leverage = 1.0