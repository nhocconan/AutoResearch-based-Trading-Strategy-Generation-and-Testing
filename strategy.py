#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe with volume confirmation and daily trend filter.
Trades reversals at key pivot levels (H3/L3) in direction of daily trend to capture mean reversion
within the trend. Uses volume spikes to confirm institutional interest at these levels.
Works in both bull and bear markets by trading with the daily trend while using mean reversion
entries. Target: 15-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Based on previous day's high, low, close
    ph = df_1d['high'].shift(1).values  # Previous day high
    pl = df_1d['low'].shift(1).values   # Previous day low
    pc = df_1d['close'].shift(1).values # Previous day close
    
    # Camarilla levels
    # H4 = close + 1.5*(high-low)
    # H3 = close + 1.1*(high-low)
    # H2 = close + 0.55*(high-low)
    # H1 = close + 0.275*(high-low)
    # L1 = close - 0.275*(high-low)
    # L2 = close - 0.55*(high-low)
    # L3 = close - 1.1*(high-low)
    # L4 = close - 1.5*(high-low)
    hl_range = ph - pl
    h3 = pc + 1.1 * hl_range
    l3 = pc - 1.1 * hl_range
    
    # Daily EMA for trend filter (20-period)
    ema_20 = df_1d['close'].ewm(span=20, adjust=False).mean().values
    
    # Align all 1d data to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or
            vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches L3 level or trend turns bearish
            if close[i] <= l3_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches H3 level or trend turns bullish
            if close[i] >= h3_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches L3 level with volume and bullish daily trend
            if (close[i] <= l3_aligned[i] * 1.001 and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches H3 level with volume and bearish daily trend
            elif (close[i] >= h3_aligned[i] * 0.999 and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals