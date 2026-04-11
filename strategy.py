#!/usr/bin/env python3
# 12h_1d_camarilla_pivot_volume_v1
# Strategy: 12h Camarilla pivot with 1d volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Camarilla levels act as strong support/resistance. Price touching L3/L4 with volume spike and bullish 1d trend triggers longs; touching H3/H4 with volume spike and bearish 1d trend triggers shorts. Low trade frequency avoids fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Camarilla pivot levels on 12h (using previous 12h bar)
    # Calculate from previous bar's high, low, close
    phigh = np.roll(high, 1)
    plow = np.roll(low, 1)
    pclose = np.roll(close, 1)
    phigh[0] = high[0]
    plow[0] = low[0]
    pclose[0] = close[0]
    
    pivot = (phigh + plow + pclose) / 3
    range_val = phigh - plow
    
    # Camarilla levels
    H4 = pivot + (range_val * 1.5 / 2)
    H3 = pivot + (range_val * 1.25 / 2)
    L3 = pivot - (range_val * 1.25 / 2)
    L4 = pivot - (range_val * 1.5 / 2)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Current 1d volume (aligned)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if np.isnan(H4[i]) or np.isnan(L4[i]) or \
           np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg_20_1d_aligned[i]) or \
           np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.5 * vol_avg_20_1d_aligned[i]
        
        # Trend filter: close vs 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Price touches L3/L4 AND uptrend AND volume confirmation
        if (close[i] <= L3[i] or close[i] <= L4[i]) and uptrend and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price touches H3/H4 AND downtrend AND volume confirmation
        elif (close[i] >= H3[i] or close[i] >= H4[i]) and downtrend and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price crosses opposite H3/L3
        elif position == 1 and close[i] >= H3[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] <= L3[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals