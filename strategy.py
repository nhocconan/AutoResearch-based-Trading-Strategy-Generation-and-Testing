#!/usr/bin/env python3
"""
Hypothesis: 1h EMA(21) + 4h Supertrend(10,3) + Volume Spike + Session Filter
- Uses 4h Supertrend for HTF trend direction (works in bull/bear via ATR adaptive bands)
- 1h EMA(21) for entry timing precision (avoid whipsaws)
- Volume spike > 2.0 * 20-period MA to confirm momentum
- Session filter 08-20 UTC to avoid low-liquidity hours
- Discrete signal size: 0.20 to minimize fee churn
- Target: 60-150 total trades over 4 years (15-37/year)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) - prices.index is DatetimeIndex
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h Supertrend for HTF trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:  # Need enough data for ATR and Supertrend
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # ATR(10) for Supertrend
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - pd.Series(close_4h).shift(1)))
    tr3 = pd.Series(np.abs(low_4h - pd.Series(close_4h).shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_10 = tr.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_4h + low_4h) / 2
    upper_band = hl2 + (3.0 * atr_10)
    lower_band = hl2 - (3.0 * atr_10)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_4h, np.nan, dtype=float)
    direction = np.full_like(close_4h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if np.isnan(atr_10[i-1]) or np.isnan(upper_band[i-1]) or np.isnan(lower_band[i-1]):
            continue
            
        # Upper band logic
        if close_4h[i-1] <= upper_band[i-1]:
            upper_band[i] = min(upper_band[i], upper_band[i-1])
        else:
            upper_band[i] = upper_band[i]
            
        # Lower band logic
        if close_4h[i-1] >= lower_band[i-1]:
            lower_band[i] = max(lower_band[i], lower_band[i-1])
        else:
            lower_band[i] = lower_band[i]
        
        # Supertrend logic
        if supertrend[i-1] == upper_band[i-1]:
            if close_4h[i] <= upper_band[i]:
                supertrend[i] = upper_band[i]
            else:
                supertrend[i] = lower_band[i]
                direction[i] = -1
        else:
            if close_4h[i] >= lower_band[i]:
                supertrend[i] = lower_band[i]
            else:
                supertrend[i] = upper_band[i]
                direction[i] = 1
    
    # Align Supertrend direction to 1h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_4h, direction.astype(float))
    
    # 1h EMA(21) for entry timing
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 21, 20)  # Need 4h Supertrend, EMA21, volume MA
    
    for i in range(start_idx, n):
        # Skip if not in trading session or data not ready
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        if (np.isnan(supertrend_direction_aligned[i]) or np.isnan(ema_21[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Supertrend uptrend AND price > EMA21 AND volume spike
            if supertrend_direction_aligned[i] == 1 and close[i] > ema_21[i] and volume_spike[i]:
                signals[i] = 0.20
                position = 1
            # Short: Supertrend downtrend AND price < EMA21 AND volume spike
            elif supertrend_direction_aligned[i] == -1 and close[i] < ema_21[i] and volume_spike[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: Supertrend turns down OR price < EMA21
            if supertrend_direction_aligned[i] == -1 or close[i] < ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: Supertrend turns up OR price > EMA21
            if supertrend_direction_aligned[i] == 1 or close[i] > ema_21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA21_4hSupertrend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0