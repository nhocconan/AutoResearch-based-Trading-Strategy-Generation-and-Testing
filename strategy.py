#!/usr/bin/env python3
name = "12h_WilliamsAlligator_ElderRay_Trend_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_williams_alligator(high, low):
    """Calculate Williams Alligator components: Jaw (13), Teeth (8), Lips (5) SMAs with forward shifts."""
    # Jaw: 13-period SMMA shifted 8 bars forward
    jaw = pd.Series(high).rolling(window=13, min_periods=13).mean().shift(8)
    # Teeth: 8-period SMMA shifted 5 bars forward
    teeth = pd.Series(low).rolling(window=8, min_periods=8).mean().shift(5)
    # Lips: 5-period SMMA shifted 3 bars forward
    lips = pd.Series(high).rolling(window=5, min_periods=5).mean().shift(3)
    return jaw.values, teeth.values, lips.values

def calculate_elder_ray(high, low, close):
    """Calculate Elder Ray: Bull Power (High - EMA13), Bear Power (Low - EMA13)."""
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean()
    bull_power = high - ema13
    bear_power = low - ema13
    return bull_power.values, bear_power.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator on 12h
    jaw, teeth, lips = calculate_williams_alligator(high, low)
    
    # Elder Ray on 12h
    bull_power, bear_power = calculate_elder_ray(high, low, close)
    
    # Volume filter: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 21)  # Alligator lips needs 5+3=8, but we use 21 for safety with shifts
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Alligator aligned (Lips > Teeth > Jaw) + Bull Power positive + above weekly EMA50 + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and bull_power[i] > 0 and close[i] > ema_50_1w_aligned[i] and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned (Lips < Teeth < Jaw) + Bear Power negative + below weekly EMA50 + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and bear_power[i] < 0 and close[i] < ema_50_1w_aligned[i] and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Alligator reverses or power diverges
            if position == 1:
                if lips[i] < teeth[i] or bull_power[i] <= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if lips[i] > teeth[i] or bear_power[i] >= 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals