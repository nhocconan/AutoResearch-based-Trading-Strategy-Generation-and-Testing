#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with 1d EMA Trend Filter and Volume Spike
Based on top performers: Uses Camarilla pivot levels from daily timeframe for structure,
1d EMA34 for trend filter, and volume spike for confirmation. Targets 20-50 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for Camarilla levels and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1, S1 (most important for intraday)
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    rang = high_1d - low_1d
    R1 = close_1d + rang * 1.1 / 12
    S1 = close_1d - rang * 1.1 / 12
    
    # Align to 4h timeframe (values update only after daily bar closes)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume / 20-period average
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio = prices['volume'].values / vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema_34_4h[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        vol_ratio_val = vol_ratio[i]
        
        if position == 0:
            # Enter long: price breaks above R1 + above EMA34 + volume spike
            if (price_close > R1_4h[i] and 
                price_close > ema_34_4h[i] and 
                vol_ratio_val > 2.0):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + below EMA34 + volume spike
            elif (price_close < S1_4h[i] and 
                  price_close < ema_34_4h[i] and 
                  vol_ratio_val > 2.0):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: price returns to opposite Camarilla level or volume drops
            if position == 1 and price_close < S1_4h[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > R1_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0