#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R3/S3 breakout with 1d EMA(34) trend filter and volume confirmation (>2.0x 20 EMA volume)
# Uses Camarilla pivot levels from prior completed 1d bar for structure (R3/S3 = fade zone, R4/S4 = breakout)
# 1d EMA(34) filter ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation ensures breakout has sufficient participation
# Discrete sizing 0.25 balances risk and return while minimizing fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe
# Works in both bull (R4/S4 breakout continuation) and bear (R4/S4 breakdown continuation) markets
# BTC/ETH focus: avoids SOL-only bias by requiring HTF trend alignment and volume confirmation

name = "6h_Camarilla_R3S3_Breakout_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA calculation
        return np.zeros(n)
    
    # Calculate 1d EMA(34) trend filter from prior completed 1d bar
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_shifted = np.roll(ema_34_1d, 1)
    ema_34_1d_shifted[0] = np.nan
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_shifted)
    
    # Calculate Camarilla pivot levels from prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    camarilla_h4 = close_1d + 1.1 * (high_1d - low_1d) / 2  # R3
    camarilla_l4 = close_1d - 1.1 * (high_1d - low_1d) / 2  # S3
    camarilla_h5 = close_1d + 1.1 * (high_1d - low_1d)      # R4
    camarilla_l5 = close_1d - 1.1 * (high_1d - low_1d)      # S4
    
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    camarilla_h4_shifted = np.roll(camarilla_h4, 1)
    camarilla_l4_shifted = np.roll(camarilla_l4, 1)
    camarilla_h5_shifted = np.roll(camarilla_h5, 1)
    camarilla_l5_shifted = np.roll(camarilla_l5, 1)
    camarilla_h4_shifted[0] = np.nan
    camarilla_l4_shifted[0] = np.nan
    camarilla_h5_shifted[0] = np.nan
    camarilla_l5_shifted[0] = np.nan
    
    # Align Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_shifted)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_shifted)
    h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5_shifted)
    l5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l5_shifted)
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(h4_aligned[i]) or 
            np.isnan(l4_aligned[i]) or np.isnan(h5_aligned[i]) or 
            np.isnan(l5_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla H5 (R4) + price > 1d EMA34 + volume spike
            if close[i] > h5_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla L5 (S4) + price < 1d EMA34 + volume spike
            elif close[i] < l5_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla H4 (R3) OR price crosses below 1d EMA34
            if close[i] < h4_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to Camarilla L4 (S3) OR price crosses above 1d EMA34
            if close[i] > l4_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals