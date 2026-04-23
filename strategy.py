#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume spike.
Long when price breaks above R1 AND price > 1d EMA34 AND volume > 2.0x average.
Short when price breaks below S1 AND price < 1d EMA34 AND volume > 2.0x average.
Exit when price crosses below R1 (long) or above S1 (short).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 15-25 trades/year per symbol.
Camarilla pivot levels provide precise intraday support/resistance, effective in ranging markets with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Camarilla calculation - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels for current 12h bar (based on previous 12h bar)
    # Camarilla R1 = close + (high - low) * 1.1/12
    # Camarilla S1 = close - (high - low) * 1.1/12
    prev_close_12h = np.roll(close_12h, 1)
    prev_high_12h = np.roll(high_12h, 1)
    prev_low_12h = np.roll(low_12h, 1)
    prev_close_12h[0] = close_12h[0]  # first bar
    prev_high_12h[0] = high_12h[0]
    prev_low_12h[0] = low_12h[0]
    
    R1 = prev_close_12h + (prev_high_12h - prev_low_12h) * 1.1 / 12
    S1 = prev_close_12h - (prev_high_12h - prev_low_12h) * 1.1 / 12
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume average (20-period) on 1d timeframe
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Use 12h close for price comparison
        price_12h = close_12h[i]
        vol_ma_val = vol_ma_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 AND price > 1d EMA34 AND volume spike
            if (price_12h > R1[i] and 
                price_12h > ema34_1d_aligned[i] and 
                volume_12h[i] > 2.0 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND price < 1d EMA34 AND volume spike
            elif (price_12h < S1[i] and 
                  price_12h < ema34_1d_aligned[i] and 
                  volume_12h[i] > 2.0 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below R1
                if price_12h < R1[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above S1
                if price_12h > S1[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Camarilla_R1S1_1dEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0