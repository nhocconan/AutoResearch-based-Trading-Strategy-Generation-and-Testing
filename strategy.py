#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Weekly VWAP + Volume Imbalance Strategy
# Uses weekly VWAP as institutional reference point and volume imbalance (buying/selling pressure)
# to detect institutional accumulation/distribution. Works in both bull and bear markets by
# following smart money flow. Target: 50-150 total trades over 4 years (~12-37/year).
# Weekly VWAP provides strong support/resistance; volume imbalance filters for genuine interest.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for VWAP calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate weekly VWAP
    vwap_1w = np.full(len(df_1w), np.nan)
    for i in range(len(df_1w)):
        tp = (high_1w[i] + low_1w[i] + close_1w[i]) / 3
        if i == 0:
            vwap_1w[i] = tp
            vol_cum = volume_1w[i]
            tpv_cum = tp * volume_1w[i]
        else:
            vol_cum += volume_1w[i]
            tpv_cum += tp * volume_1w[i]
            if vol_cum > 0:
                vwap_1w[i] = tpv_cum / vol_cum
            else:
                vwap_1w[i] = vwap_1w[i-1] if i > 0 else tp
    
    # Align weekly VWAP to 6h timeframe (wait for weekly close)
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    
    # Volume imbalance: buying pressure vs selling pressure over 4 periods (1 day of 6h bars)
    vol_imb = np.full(n, np.nan)
    for i in range(3, n):
        # Buy volume: when close > open
        buy_vol = 0
        sell_vol = 0
        for j in range(i-3, i+1):
            if close[j] >= prices['open'].iloc[j]:
                buy_vol += volume[j]
            else:
                sell_vol += volume[j]
        total_vol = buy_vol + sell_vol
        if total_vol > 0:
            vol_imb[i] = (buy_vol - sell_vol) / total_vol  # -1 to +1
        else:
            vol_imb[i] = 0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need weekly VWAP (1 bar) and volume imbalance (3 bars)
    start_idx = max(1, 3)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(vwap_1w_aligned[i]) or np.isnan(vol_imb[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vwap = vwap_1w_aligned[i]
        imb = vol_imb[i]
        
        # Entry conditions with hysteresis to prevent whipsaw
        if position == 0:
            # Long: price above VWAP with strong buying pressure
            if price > vwap * 1.002 and imb > 0.3:  # 0.2% buffer + buying pressure
                signals[i] = size
                position = 1
            # Short: price below VWAP with strong selling pressure
            elif price < vwap * 0.998 and imb < -0.3:  # 0.2% buffer + selling pressure
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to VWAP or selling pressure emerges
            if price <= vwap or imb < -0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to VWAP or buying pressure emerges
            if price >= vwap or imb > 0.1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Weekly_VWAP_Volume_Imbalance"
timeframe = "6h"
leverage = 1.0