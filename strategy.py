#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA34 trend filter + volume spike.
- Primary timeframe: 1d for execution, HTF: 1w for EMA34 trend filter.
- Entry: price breaks above Donchian(20) upper band (long) or below Donchian(20) lower band (short) on 1d close, with volume > 2.0x 20-period volume MA.
- Direction filter: only long when 1d close > 1w EMA34, only short when 1d close < 1w EMA34.
- Donchian breakout captures momentum; EMA34 filter avoids counter-trend trades.
- Volume confirmation reduces false signals.
- Exit: opposite Donchian band touch (long exits at lower band, short exits at upper band) or trend filter reversal.
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
- Works in bull via buying breakouts in uptrend, in bear via selling breakdowns in downtrend.
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
    
    # Calculate 1w EMA34 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) bands on 1d data
    if len(high) < 20:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # Need 1w EMA34, Donchian(20), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band with volume spike AND uptrend (close > 1w EMA34)
            if (close[i] > highest_high[i] and volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band with volume spike AND downtrend (close < 1w EMA34)
            elif (close[i] < lowest_low[i] and volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns below Donchian lower band or trend reversal
            if close[i] < lowest_low[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns above Donchian upper band or trend reversal
            if close[i] > highest_high[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0