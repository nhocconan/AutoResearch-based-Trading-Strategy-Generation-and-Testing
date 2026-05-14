#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla H3/L3 Breakout with 12h EMA34 Trend Filter and Volume Spike.
- Primary timeframe: 4h for execution, HTF: 12h for EMA34 trend filter.
- Entry: 4h close breaks above H3 or below L3 from prior 1d Camarilla calculation + volume spike (>2.0x 20-period volume MA).
- Direction filter: only long when 4h close > 12h EMA34, only short when 4h close < 12h EMA34.
- Volume confirmation reduces false breakouts; Camarilla H3/L3 provides meaningful support/resistance.
- Exit: opposite Camarilla level touch (long exits at L3, short exits at H3) or trend filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via breakout continuation, in bear via mean reversion at extreme levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate prior 1d Camarilla levels (H3, L3)
    # Need true 1d OHLC for proper Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    # Use prior day's OHLC (shift by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # Handle first value (no prior day)
    prev_high[0] = prev_close[0]  # fallback
    prev_low[0] = prev_close[0]
    prev_close[0] = close_1d[0] if len(close_1d) > 0 else 0
    
    # Calculate Camarilla H3 and L3 levels
    hl_range = prev_high - prev_low
    h3 = prev_close + hl_range * 1.1 / 4
    l3 = prev_close - hl_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 1)  # Need 12h EMA34, volume MA, and prior 1d Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above H3 with volume spike AND uptrend (close > 12h EMA34)
            if (close[i] > h3_aligned[i] and 
                close[i] > ema_34_12h_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with volume spike AND downtrend (close < 12h EMA34)
            elif (close[i] < l3_aligned[i] and 
                  close[i] < ema_34_12h_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below L3 (mean reversion) or trend reversal
            if close[i] < l3_aligned[i] or close[i] < ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above H3 (mean reversion) or trend reversal
            if close[i] > h3_aligned[i] or close[i] > ema_34_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_H3L3_VolumeSpike_12hEMA34_v1"
timeframe = "4h"
leverage = 1.0