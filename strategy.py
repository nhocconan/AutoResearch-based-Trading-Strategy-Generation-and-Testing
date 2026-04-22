#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot point breakout with 1d EMA34 trend filter and volume spike confirmation.
# Camarilla pivot levels (R1, R2, S1, S2) derived from previous day's high/low/close provide institutional support/resistance.
# Breakout above R1 or below S1 with volume spike (>2x 20-period average) and trend alignment (price > 1d EMA34 for longs, < for shorts)
# captures breakout moves while avoiding false signals. Designed for low trade frequency (~20-40/year) to minimize fee decay.
# Works in both bull and bear markets by following higher timeframe trend and requiring volume confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for Camarilla pivot calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    r2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    s2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    
    # Calculate 34-period EMA on 1d close for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 4h timeframe (waits for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (strict filter for low frequency)
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above R1 + uptrend + volume spike
            if price > r1_val and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below S1 + downtrend + volume spike
            elif price < s1_val and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price breaks below S1 or trend breaks
                if price < s1_val or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price breaks above R1 or trend breaks
                if price > r1_val or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0