#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeConfirm_Regime
Hypothesis: Trade 4h timeframe using Camarilla pivot levels (R1, S1) from prior day for entry, 
12h EMA34 for trend filter, volume spike (>1.8x 20-bar MA) for confirmation, and 
choppiness regime filter (CHOP < 50 for trending markets). 
Enter long when price breaks above Camarilla R1 AND above 12h EMA34 AND volume spike AND chop < 50. 
Enter short when price breaks below Camarilla S3 AND below 12h EMA34 AND volume spike AND chop < 50. 
Exit on opposite Camarilla touch (S1 for long, R1 for short) or trend reversal (close crosses 12h EMA34). 
Uses discrete sizing 0.30 to balance return and drawdown. Target 20-50 trades/year on 4h timeframe. 
Camarilla R1/S1 levels provide breakout points with moderate filtering. 
The 12h EMA34 filter ensures we trade with the intermediate trend, improving performance in both bull and bear markets. 
Volume confirmation avoids breakouts from low participation. 
Choppiness regime filter avoids whipsaws in sideways markets. 
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Get 1d data for Camarilla pivot levels (prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for prior day: R1, S1
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_r1_1d = close_1d + (1.1 * (high_1d - low_1d) / 12)
    camarilla_s1_1d = close_1d - (1.1 * (high_1d - low_1d) / 12)
    
    # Align Camarilla levels to 4h timeframe (prior day's levels available at 00:00 UTC)
    camarilla_r1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Calculate 20-bar volume MA on 4h for volume spike detection
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.8 * vol_ma_4h)
    
    # Calculate Choppiness Index on 4h (using high, low, close)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low) / log10(14)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first bar
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_14 * 14) / np.log10(highest_high_14 - lowest_low_14)
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)  # avoid division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 12h EMA34 (34), 1d Camarilla (0), 4h volume MA (20), 4h CHOP (14)
    start_idx = max(34, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r1_1d_aligned[i]) or
            np.isnan(camarilla_s1_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 AND above 12h EMA34 AND volume spike AND chop < 50
            long_setup = (close[i] > camarilla_r1_1d_aligned[i]) and \
                         (close[i] > ema_34_12h_aligned[i]) and \
                         volume_spike_4h[i] and \
                         (chop[i] < 50)
            # Short: price breaks below Camarilla S1 AND below 12h EMA34 AND volume spike AND chop < 50
            short_setup = (close[i] < camarilla_s1_1d_aligned[i]) and \
                          (close[i] < ema_34_12h_aligned[i]) and \
                          volume_spike_4h[i] and \
                          (chop[i] < 50)
            
            if long_setup:
                signals[i] = 0.30
                position = 1
            elif short_setup:
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Exit: price touches Camarilla S1 OR closes below 12h EMA34
            if (close[i] <= camarilla_s1_1d_aligned[i]) or \
               (close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Exit: price touches Camarilla R1 OR closes above 12h EMA34
            if (close[i] >= camarilla_r1_1d_aligned[i]) or \
               (close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA34_VolumeConfirm_Regime"
timeframe = "4h"
leverage = 1.0