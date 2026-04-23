#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND close > 1d EMA34 AND volume > 1.8x average.
Short when price breaks below Camarilla S1 AND close < 1d EMA34 AND volume > 1.8x average.
Exit when price reverses to Camarilla pivot point (PP) OR volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Camarilla levels from 1d provide strong intraday support/resistance that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Camarilla levels and EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on 1d data for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: PP = (H+L+C)/3, R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    PP_1d = (high_1d + low_1d + close_1d) / 3.0
    R1_1d = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    S1_1d = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align 1d Camarilla levels to 4h timeframe
    PP_1d_aligned = align_htf_to_ltf(prices, df_1d, PP_1d)
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(PP_1d_aligned[i]) or 
            np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        PP = PP_1d_aligned[i]
        R1 = R1_1d_aligned[i]
        S1 = S1_1d_aligned[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: break above R1 AND price > 1d EMA34 AND volume spike
            if (price > R1 and price > ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 AND price < 1d EMA34 AND volume spike
            elif (price < S1 and price < ema34_val and vol_current > 1.8 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to pivot point OR volume drops below average
                if (price <= PP or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to pivot point OR volume drops below average
                if (price >= PP or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_R1S1_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0