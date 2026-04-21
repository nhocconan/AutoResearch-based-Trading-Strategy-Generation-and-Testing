#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_VolumeFilter_Session
Hypothesis: Trade Camarilla R1/S1 breakouts on 1h with 4h trend filter, volume confirmation, and session filter (08-20 UTC).
In uptrend (4h close > EMA34): buy breakouts above R1. In downtrend: sell breakdowns below S1.
Volume filter ensures breakout legitimacy. Session filter avoids low-liquidity hours.
Target: 15-37 trades/year per symbol (60-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once for trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous 4h bar's OHLC for Camarilla calculation
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1 (primary breakout levels)
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Precompute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        if position == 0:
            # Long conditions: price > R1 AND 4h uptrend AND volume
            if (price > r1_aligned[i] and 
                ema_34_4h_aligned[i] > ema_34_4h_aligned[i-1] and  # 4h EMA rising
                volume_ok):
                signals[i] = 0.20
                position = 1
            # Short conditions: price < S1 AND 4h downtrend AND volume
            elif (price < s1_aligned[i] and 
                  ema_34_4h_aligned[i] < ema_34_4h_aligned[i-1] and  # 4h EMA falling
                  volume_ok):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price < 4h EMA34 (trend reversal) or price < S1 (mean reversion)
            if (price < ema_34_4h_aligned[i] or 
                price < s1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price > 4h EMA34 (trend reversal) or price > R1 (mean reversion)
            if (price > ema_34_4h_aligned[i] or 
                price > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0