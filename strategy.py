#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1D Camarilla Pivot Breakout with Volume Confirmation and ADX Trend Filter
# Uses daily Camarilla levels (H4, L4, H3, L3) as institutional support/resistance
# Breakout above H4 or below L4 with volume > 1.5x average and ADX > 25 (trending)
# Works in bull/bear as Camarilla adapts to volatility and ADX filters range
# Target: 20-35 trades/year per symbol (80-140 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for Camarilla pivot and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla levels from previous day
    # H4 = Close + 1.1*(High - Low)
    # L4 = Close - 1.1*(High - Low)
    # H3 = Close + 1.1*(High - Low)/2
    # L3 = Close - 1.1*(High - Low)/2
    if len(df_1d) < 2:
        return np.zeros(n)
    
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    camarilla_h4 = prev_close + 1.1 * (prev_high - prev_low)
    camarilla_l4 = prev_close - 1.1 * (prev_high - prev_low)
    camarilla_h3 = prev_close + 1.1 * (prev_high - prev_low) / 2
    camarilla_l3 = prev_close - 1.1 * (prev_high - prev_low) / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # ADX (14) on 1d for trend strength
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Directional Movement
    up = df_1d['high'] - df_1d['high'].shift(1)
    down = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    plus_dm14 = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum()
    minus_dm14 = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * plus_dm14 / tr14
    minus_di = 100 * minus_dm14 / tr14
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_values)
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(100, 20, 14)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(adx_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates trending market
        trending = adx_aligned[i] > 25
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Enter long: break above H4 with volume and trend
            if (close[i] > camarilla_h4_aligned[i] and 
                volume_confirmed and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: break below L4 with volume and trend
            elif (close[i] < camarilla_l4_aligned[i] and 
                  volume_confirmed and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H3 or breaks below L3
            if close[i] < camarilla_h3_aligned[i] or close[i] < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to L3 or breaks above H3
            if close[i] > camarilla_l3_aligned[i] or close[i] > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_ADX_v1"
timeframe = "4h"
leverage = 1.0