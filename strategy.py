#!/usr/bin/env python3
# Hypothesis: 4h timeframe with 12h Supertrend for trend direction and volume confirmation for entry.
# Uses Supertrend to identify market regime: long when price > Supertrend, short when price < Supertrend.
# Enters only on volume spikes (volume > 1.5x 20-period volume average) during pullbacks to EMA21.
# Exits when price crosses Supertrend in opposite direction or volume drops below average.
# Target: 20-50 trades per year with position size 0.25 to manage drawdown and fees.

name = "4h_Supertrend_Volume_Pullback_12h"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 10:
        return np.zeros(n)
    
    # Calculate Supertrend on 12h data
    atr_period = 10
    multiplier = 3.0
    
    hl2 = (df_12h['high'] + df_12h['low']) / 2
    atr = pd.Series(index=df_12h.index, dtype=float)
    tr = pd.Series(index=df_12h.index, dtype=float)
    
    for i in range(len(df_12h)):
        if i == 0:
            tr.iloc[i] = df_12h['high'].iloc[i] - df_12h['low'].iloc[i]
        else:
            tr.iloc[i] = max(
                df_12h['high'].iloc[i] - df_12h['low'].iloc[i],
                abs(df_12h['high'].iloc[i] - df_12h['close'].iloc[i-1]),
                abs(df_12h['low'].iloc[i] - df_12h['close'].iloc[i-1])
            )
        if i < atr_period:
            atr.iloc[i] = tr.iloc[:i+1].mean()
        else:
            atr.iloc[i] = (atr.iloc[i-1] * (atr_period-1) + tr.iloc[i]) / atr_period
    
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    
    supertrend = pd.Series(index=df_12h.index, dtype=float)
    direction = pd.Series(index=df_12h.index, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(len(df_12h)):
        if i == 0:
            supertrend.iloc[i] = lowerband.iloc[i]
            direction.iloc[i] = 1
        else:
            if close_12h := df_12h['close'].iloc[i] if 'close_12h' in locals() else df_12h['close'].iloc[i]:
                pass
            close_12h = df_12h['close'].iloc[i]
            
            if i < atr_period:
                supertrend.iloc[i] = lowerband.iloc[i]
                direction.iloc[i] = 1
            else:
                if supertrend.iloc[i-1] == upperband.iloc[i-1]:
                    if close_12h <= upperband.iloc[i]:
                        supertrend.iloc[i] = upperband.iloc[i]
                    else:
                        supertrend.iloc[i] = lowerband.iloc[i]
                        direction.iloc[i] = -1
                else:
                    if close_12h >= lowerband.iloc[i]:
                        supertrend.iloc[i] = lowerband.iloc[i]
                        direction.iloc[i] = -1
                    else:
                        supertrend.iloc[i] = upperband.iloc[i]
                        direction.iloc[i] = 1
    
    # Align Supertrend and direction to 4h timeframe
    supertrend_values = supertrend.values
    direction_values = direction.values
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend_values)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction_values)
    
    # Calculate EMA21 on 4h for pullback entries
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:
        return np.zeros(n)
    
    close_4h = df_4h['close']
    ema_21 = close_4h.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_4h, ema_21)
    
    # Volume spike detector: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(supertrend_aligned[i]) or
            np.isnan(direction_aligned[i]) or
            np.isnan(ema_21_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: uptrend (direction=1), price near EMA21 pullback, volume spike
            if (direction_aligned[i] == 1 and 
                close[i] >= ema_21_aligned[i] * 0.98 and  # Allow small deviation from EMA
                close[i] <= ema_21_aligned[i] * 1.02 and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: downtrend (direction=-1), price near EMA21 pullback, volume spike
            elif (direction_aligned[i] == -1 and 
                  close[i] >= ema_21_aligned[i] * 0.98 and
                  close[i] <= ema_21_aligned[i] * 1.02 and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reverses (direction=-1) or volume drops
            if direction_aligned[i] == -1 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reverses (direction=1) or volume drops
            if direction_aligned[i] == 1 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals