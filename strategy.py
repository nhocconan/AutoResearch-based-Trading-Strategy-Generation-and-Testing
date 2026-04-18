#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1
Camarilla pivot breakout on 12h with volume confirmation and 1w trend filter:
- Long when price breaks above R1 with volume > 1.5x 20-period average
- Short when price breaks below S1 with volume > 1.5x 20-period average
- Trend filter: 1w EMA34 - only long when price > EMA34, short when price < EMA34
- Designed for 12-30 trades/year per symbol
Works in bull markets (breakouts with trend) and bear markets (breakdowns with trend)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays."""
    n = len(high)
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    PP = np.full(n, np.nan)
    
    for i in range(n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]):
            continue
        PP[i] = (high[i] + low[i] + close[i]) / 3
        R1[i] = close[i] + (high[i] - low[i]) * 1.1 / 12
        S1[i] = close[i] - (high[i] - low[i]) * 1.1 / 12
    
    return PP, R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34
    close_1w_series = pd.Series(close_1w)
    ema_34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP_1d, R1_1d, S1_1d = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align daily Camarilla to 12h timeframe
    R1_1d_aligned = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_aligned = align_htf_to_ltf(prices, df_1d, S1_1d)
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need 20 for volume MA + buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(R1_1d_aligned[i]) or np.isnan(S1_1d_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price relative to weekly EMA34
        price_above_ema = close[i] > ema_34_1w_aligned[i]
        price_below_ema = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume + trend
            if close[i] > R1_1d_aligned[i] and volume_ok and price_above_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 + volume + trend
            elif close[i] < S1_1d_aligned[i] and volume_ok and price_below_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S1 (reverse) or trend change
            if close[i] < S1_1d_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R1 (reverse) or trend change
            if close[i] > R1_1d_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0