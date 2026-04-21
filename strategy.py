#!/usr/bin/env python3
"""
1d_Camarilla_Pivot_Breakout_Volume_ATR_v2
Hypothesis: Breakout of Camarilla R4/S4 levels on 1d timeframe with 1w trend filter (EMA34) and volume confirmation.
Works in bull/bear: In uptrend, buy R4 breakout; in downtrend, sell S4 breakout. Uses 1w EMA for trend filter.
Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R4, S4 (breakout levels)
    rang = prev_high - prev_low
    r4 = prev_close + rang * 6.0 / 12
    s4 = prev_close - rang * 6.0 / 12
    
    # Align to 1d timeframe (no shift needed as we use previous day's levels)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Load 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Determine trend: weekly EMA34 rising/falling
        if i >= 1:
            weekly_uptrend = ema_34_1w_aligned[i] > ema_34_1w_aligned[i-1]
            weekly_downtrend = ema_34_1w_aligned[i] < ema_34_1w_aligned[i-1]
        else:
            weekly_uptrend = True
            weekly_downtrend = False
        
        if position == 0:
            # Long: price breaks above R4 AND weekly uptrend AND volume confirmation
            if (price > r4_aligned[i] and 
                weekly_uptrend and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S4 AND weekly downtrend AND volume confirmation
            elif (price < s4_aligned[i] and 
                  weekly_downtrend and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 1w EMA34 (trend reversal)
            if price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 1w EMA34 (trend reversal)
            if price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_Pivot_Breakout_Volume_ATR_v2"
timeframe = "1d"
leverage = 1.0