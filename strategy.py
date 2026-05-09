#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WilliamsAlligator_ElderRay_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Williams Alligator on 1d: SMA(13,8), SMA(8,5), SMA(5,3)
    sm3 = pd.Series(close).rolling(window=3, min_periods=3).mean().shift(2)  # Jaw (13)
    sm5 = pd.Series(close).rolling(window=5, min_periods=5).mean().shift(3)  # Teeth (8)
    sm8 = pd.Series(close).rolling(window=8, min_periods=8).mean().shift(5)  # Lips (5)
    
    # Elder Ray on 1d: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1w EMA34 for trend filter
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume filter: above 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(sm3[i]) or np.isnan(sm5[i]) or np.isnan(sm8[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma[i]  # Volume confirmation
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend
        alligator_up = sm8[i] > sm5[i] and sm5[i] > sm3[i]
        alligator_down = sm8[i] < sm5[i] and sm5[i] < sm3[i]
        
        if position == 0:
            # Long: Alligator up + Bull Power > 0 + above 1w EMA + volume
            if (alligator_up and 
                bull_power[i] > 0 and 
                close[i] > ema_1w_aligned[i] and 
                vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: Alligator down + Bear Power < 0 + below 1w EMA + volume
            elif (alligator_down and 
                  bear_power[i] < 0 and 
                  close[i] < ema_1w_aligned[i] and 
                  vol_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator turns down OR Bull Power turns negative
            if not alligator_up or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator turns up OR Bear Power turns positive
            if not alligator_down or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals