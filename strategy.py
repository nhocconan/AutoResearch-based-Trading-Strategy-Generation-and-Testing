#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot point reversal with 1d trend filter and volume confirmation.
# Uses daily pivot levels (S1, R1) from previous day for mean-reversion entries.
# Only takes longs when price > 1d EMA34 (uptrend filter) and shorts when price < 1d EMA34 (downtrend filter).
# Volume confirmation requires current volume > 2.0x 20-period average to avoid false signals.
# Designed for low-frequency, high-probability reversals in ranging markets.
# Targets 15-25 trades/year with strict entry conditions to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for pivot points and EMA trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use S1 and R1: S1 = close - 1.05*(high-low), R1 = close + 1.05*(high-low)
    rng = high_1d - low_1d
    S1 = close_1d - 1.05 * rng
    R1 = close_1d + 1.05 * rng
    
    # Calculate 34-period EMA on 1d data for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d indicators to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period average volume for volume spike detection
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(S1_aligned[i]) or 
            np.isnan(R1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        s1 = S1_aligned[i]
        r1 = R1_aligned[i]
        ema_val = ema_34_aligned[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        vol_spike = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long conditions: price at S1 support + uptrend + volume spike
            if price <= s1 and price > ema_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short conditions: price at R1 resistance + downtrend + volume spike
            elif price >= r1 and price < ema_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price reaches opposite pivot level
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price reaches R1 resistance
                if price >= r1:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price reaches S1 support
                if price <= s1:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Camarilla_S1R1_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0