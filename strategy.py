#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Trend Following with 4h Supertrend and 1d Volume Filter
# Uses 4h Supertrend (ATR=10, mult=3) for trend direction, enters on 1h pullbacks to EMA20
# Only trades during 08-20 UTC session to avoid low-liquidity hours
# Volume filter: requires 1h volume > 1.5x 20-period average to confirm conviction
# Fixed position size of 0.20 to manage risk. Designed for 15-30 trades/year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data for Supertrend (ONCE before loop)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h ATR(10)
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate 4h Supertrend
    upper_band = (high_4h + low_4h) / 2 + 3.0 * atr_10
    lower_band = (high_4h + low_4h) / 2 - 3.0 * atr_10
    
    supertrend = np.full_like(close_4h, np.nan, dtype=float)
    direction = np.full_like(close_4h, 1, dtype=int)  # 1 = uptrend, -1 = downtrend
    
    for i in range(10, len(close_4h)):
        if np.isnan(atr_10[i]):
            continue
            
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
    
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend to 1h
    supertrend_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_4h, direction.astype(float))
    
    # Load 1d data for volume filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_volume_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_ratio_1d = volume_1d / avg_volume_1d
    volume_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_ratio_1d)
    
    # 1h EMA(20) for pullback entries
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if data not ready or outside session
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(volume_ratio_1d_aligned[i]) or np.isnan(ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 4h uptrend + 1h price near EMA20 + strong 1d volume
            if (direction_aligned[i] == 1 and 
                close[i] > ema_20[i] * 0.995 and 
                close[i] < ema_20[i] * 1.005 and
                volume_ratio_1d_aligned[i] > 1.5):
                signals[i] = 0.20
                position = 1
            # Short: 4h downtrend + 1h price near EMA20 + strong 1d volume
            elif (direction_aligned[i] == -1 and 
                  close[i] < ema_20[i] * 1.005 and 
                  close[i] > ema_20[i] * 0.995 and
                  volume_ratio_1d_aligned[i] > 1.5):
                signals[i] = -0.20
                position = -1
        else:
            # Exit: 4h trend reversal or price moves 1.5% away from EMA20
            if position == 1:
                if (direction_aligned[i] == -1 or 
                    close[i] < ema_20[i] * 0.985 or
                    close[i] > ema_20[i] * 1.015):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            else:  # position == -1
                if (direction_aligned[i] == 1 or 
                    close[i] > ema_20[i] * 1.015 or
                    close[i] < ema_20[i] * 0.985):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals

name = "1h_Supertrend4h_EMA20Pullback_Volume1d"
timeframe = "1h"
leverage = 1.0