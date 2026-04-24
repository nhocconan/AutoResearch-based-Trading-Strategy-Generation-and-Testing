#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA34 trend filter.
- Entry: Price breaks above Donchian upper band (long) or below lower band (short) on 1d close, with volume > 1.8x 20-period volume MA.
- Direction filter: only long when 1d close > 1w EMA34, only short when 1d close < 1w EMA34.
- Donchian bands provide clear breakout levels; 1w EMA34 ensures alignment with weekly trend.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to Donchian middle band (mean of upper/lower) or trend filter reversal.
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
    
    # Calculate Donchian channels (20-period) from 1d data
    # Use rolling window on daily high/low
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    upper_band = high_ma
    lower_band = low_ma
    middle_band = (upper_band + lower_band) / 2
    
    # Volume confirmation: current volume > 1.8 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need 1w EMA34 and Donchian(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian upper band with volume spike AND uptrend (close > 1w EMA34)
            if (close[i] > upper_band[i] and volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower band with volume spike AND downtrend (close < 1w EMA34)
            elif (close[i] < lower_band[i] and volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian middle band or trend reversal
            if (close[i] < middle_band[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian middle band or trend reversal
            if (close[i] > middle_band[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA34_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0