#!/usr/bin/env python3
"""
Hypothesis: 1h timeframe with 4h Camarilla pivot breakout and volume confirmation.
Trade breakouts of Camarilla R1/S1 levels with volume spike (>1.8x 20-period average).
Use 1d ADX > 20 to filter for trending markets and avoid false breakouts in ranging conditions.
In bull markets: buy breakouts above R1; sell breakdowns below S1.
In bear markets: sell breakdowns below S1; buy breakouts above R1 (mean reversion within downtrend).
Position sizing: discrete 0.20 for entries, 0 for exits.
Target: 80-120 total trades over 4 years (20-30/year) to balance opportunity and fee drag.
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
    
    # Get 4h data for Camarilla pivots
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels (R1, S1)
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rng_4h = high_4h - low_4h
    camarilla_r1 = close_4h + 1.1 * rng_4h / 12
    camarilla_s1 = close_4h - 1.1 * rng_4h / 12
    
    # Get 1d data for ADX filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX (14)
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1h
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from ADX components
        plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        plus_dm = np.concatenate([[0], plus_dm])
        minus_dm = np.concatenate([[0], minus_dm])
        
        tr1 = high_1d - low_1d
        tr2 = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
        tr3 = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
        
        plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di)
        minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di)
        
        if np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]):
            signals[i] = 0.0
            continue
            
        uptrend = plus_di_aligned[i] > minus_di_aligned[i]
        downtrend = plus_di_aligned[i] < minus_di_aligned[i]
        strong_trend = adx_aligned[i] > 20
        
        if position == 0:
            # Long: price breaks above R1, volume spike, strong trend
            if (close[i] > camarilla_r1_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.8 and 
                strong_trend and uptrend):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1, volume spike, strong trend
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.8 and 
                  strong_trend and downtrend):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint (mean reversion) or trend weakens
            midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
            if close[i] < midpoint or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price returns to midpoint or trend weakens
            midpoint = (camarilla_r1_aligned[i] + camarilla_s1_aligned[i]) / 2
            if close[i] > midpoint or adx_aligned[i] < 15:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "1h"
leverage = 1.0