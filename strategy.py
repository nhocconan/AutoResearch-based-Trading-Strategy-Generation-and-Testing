#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA34 trend filter.
- Entry: Price breaks above Donchian(20) high (long) or below Donchian(20) low (short) on 1d close, with volume > 1.5x 20-period volume MA.
- Direction filter: only long when 1d close > 1w EMA34, only short when 1d close < 1w EMA34.
- Donchian channels provide clear breakout levels; EMA34 filters for higher timeframe trend alignment.
- Volume confirmation reduces false breakouts.
- Exit: Price returns to Donchian(20) midpoint or trend filter reversal.
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
    
    # Calculate Donchian(20) channels: use 20-period high/low
    donchian_period = 20
    high_roll = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    low_roll = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    donchian_high = high_roll
    donchian_low = low_roll
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, donchian_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike AND uptrend (close > 1w EMA34)
            if (close[i] > donchian_high[i] and volume_spike[i] and 
                close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike AND downtrend (close < 1w EMA34)
            elif (close[i] < donchian_low[i] and volume_spike[i] and 
                  close[i] < ema_34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price returns to Donchian midpoint or trend reversal
            if (close[i] < donchian_mid[i] or close[i] < ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price returns to Donchian midpoint or trend reversal
            if (close[i] > donchian_mid[i] or close[i] > ema_34_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_EMA34_Trend_VolumeSpike_v1"
timeframe = "1d"
leverage = 1.0