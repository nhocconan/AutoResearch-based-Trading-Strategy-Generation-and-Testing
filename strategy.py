#!/usr/bin/env python3
"""
4h_12h_Camarilla_Breakout_Volume
Hypothesis: Trade breakouts from 12h Camarilla pivot levels (H4/L4) on 4h timeframe with volume confirmation and ADX trend filter.
Uses 12h Camarilla levels as strong support/resistance that institutions respect. Works in bull (breakouts above H4) and bear (breakdowns below L4) markets.
Volume filter ensures institutional participation. ADX filter ensures trades align with trend (ADX > 25) to avoid whipsaw in ranging markets.
Target: 20-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    range_val = high - low
    range_val = np.where(range_val == 0, 1e-10, range_val)
    C = close
    H = high
    L = low
    R1 = C + ((H - L) * 1.0833)
    R2 = C + ((H - L) * 1.1666)
    R3 = C + ((H - L) * 1.2500)
    R4 = C + ((H - L) * 1.5000)
    S1 = C - ((H - L) * 1.0833)
    S2 = C - ((H - L) * 1.1666)
    S3 = C - ((H - L) * 1.2500)
    S4 = C - ((H - L) * 1.5000)
    return R1, R2, R3, R4, S1, S2, S3, S4

def calculate_adx(high, low, close, period=14):
    """Calculate ADX (Average Directional Index)."""
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # True Range
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    dm_plus = high - high.shift()
    dm_minus = low.shift() - low
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    # Smoothed values
    tr_ma = tr.rolling(window=period, min_periods=period).mean()
    dm_plus_ma = dm_plus.rolling(window=period, min_periods=period).mean()
    dm_minus_ma = dm_minus.rolling(window=period, min_periods=period).mean()
    
    # Directional Indicators
    di_plus = 100 * dm_plus_ma / tr_ma
    di_minus = 100 * dm_minus_ma / tr_ma
    
    # DX and ADX
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period, min_periods=period).mean()
    
    return adx.values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla levels and ADX
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h Camarilla levels
    R1_12h, R2_12h, R3_12h, R4_12h, S1_12h, S2_12h, S3_12h, S4_12h = calculate_camarilla(high_12h, low_12h, close_12h)
    
    # Calculate 12h ADX
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    
    # Align all data to 4h timeframe
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, R4_12h)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, S4_12h)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_expansion = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(R4_12h_aligned[i]) or np.isnan(S4_12h_aligned[i]) or 
            np.isnan(adx_12h_aligned[i]) or np.isnan(volume_expansion[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade when ADX > 25 (trending market)
        trending = adx_12h_aligned[i] > 25
        
        # Long: breakout above R4 with volume expansion and trending market
        long_condition = (close[i] > R4_12h_aligned[i]) and volume_expansion[i] and trending
        
        # Short: breakdown below S4 with volume expansion and trending market
        short_condition = (close[i] < S4_12h_aligned[i]) and volume_expansion[i] and trending
        
        if long_condition and position != 1:
            position = 1
            signals[i] = position_size
        elif short_condition and position != -1:
            position = -1
            signals[i] = -position_size
        else:
            # Hold current position
            signals[i] = position_size if position == 1 else (-position_size if position == -1 else 0.0)
    
    return signals

name = "4h_12h_Camarilla_Breakout_Volume"
timeframe = "4h"
leverage = 1.0