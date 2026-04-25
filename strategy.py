#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeFilter
Hypothesis: On 1d timeframe, Camarilla R1/S1 breakouts with 1w EMA34 trend filter and volume confirmation captures institutional breakouts with low trade frequency (target: 30-100 total trades over 4 years). The 1w trend filter ensures alignment with long-term momentum, reducing false breakouts in choppy markets. Volume confirmation ensures participation. Designed to work in both bull and bear markets by taking breakouts in the direction of the 1w trend. Uses discrete position sizing (0.25) to minimize fee churn.
"""

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
    
    # Get 1w data for HTF trend and Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate EMA34 on 1w for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from previous 1w bar (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    # Use previous completed 1w bar to avoid look-ahead
    prev_close = np.concatenate([[np.nan], close_1w[:-1]])
    prev_high = np.concatenate([[np.nan], high_1w[:-1]])
    prev_low = np.concatenate([[np.nan], low_1w[:-1]])
    
    camarilla_range = prev_high - prev_low
    r1 = prev_close + 1.1 * camarilla_range * 1.1 / 12
    s1 = prev_close - 1.1 * camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Volume average (20-period) for volume spike filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need warmup for calculations
    start_idx = max(50, 20)  # EMA34, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            # Hold current position or flat
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Get aligned values
        ema_val = ema_34_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_val = volume[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        
        # Volume spike condition: current volume > 1.5x 20-period average
        volume_spike = vol_val > 1.5 * vol_ma_val
        
        if position == 0:
            # Look for entry signals: Camarilla R1/S1 breakout with trend and volume
            # Long: price breaks above R1 with uptrend (close > EMA34) and volume spike
            long_signal = (high_val > r1_val) and (close_val > ema_val) and volume_spike
            # Short: price breaks below S1 with downtrend (close < EMA34) and volume spike
            short_signal = (low_val < s1_val) and (close_val < ema_val) and volume_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks below S1 (exit long)
            if close_val < s1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions:
            # 1. Opposite breakout: price breaks above R1 (exit short)
            if close_val > r1_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_1wEMA34_Trend_VolumeFilter"
timeframe = "1d"
leverage = 1.0