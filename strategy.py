#!/usr/bin/env python3

"""
Hypothesis: 6-hour ADX-based trend strength with 12-hour Williams %R mean reversion and volume confirmation.
Trades long when ADX > 25 (trending) and Williams %R < -80 (oversold) in uptrend context.
Trades short when ADX > 25 and Williams %R > -20 (overbought) in downtrend context.
Uses volume spike (1.5x 20-period average) to confirm momentum.
Designed for 60-120 total trades over 4 years (15-30/year) with clear trend/momentum alignment.
Works in bull markets via trend-following oversold bounces and in bear markets via trend-following overbought fades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for ADX and Williams %R - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX (14-period)
    plus_dm = np.zeros_like(high_12h)
    minus_dm = np.zeros_like(low_12h)
    plus_dm[1:] = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                           np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    minus_dm[1:] = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                            np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = tr2[0] = tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean()
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum() / atr.replace(0, np.nan)
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum() / atr.replace(0, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high_12h).rolling(window=14, min_periods=14).max()
    lowest_low = pd.Series(low_12h).rolling(window=14, min_periods=14).min()
    williams_r = -100 * (highest_high - close_12h) / (highest_high - lowest_low).replace(0, np.nan)
    williams_r = williams_r.values
    
    # Align ADX and Williams %R
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    williams_r_aligned = align_htf_to_ltf(prices, df_12h, williams_r)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: ADX > 25 (trending) and Williams %R < -80 (oversold)
            if adx_aligned[i] > 25 and williams_r_aligned[i] < -80:
                signals[i] = 0.25
                position = 1
            # Short: ADX > 25 (trending) and Williams %R > -20 (overbought)
            elif adx_aligned[i] > 25 and williams_r_aligned[i] > -20:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: ADX < 20 (trend weakening) or Williams %R reverts to mid-range (-50)
            exit_signal = False
            
            if position == 1:
                # Exit long: trend weakening or oversold condition resolved
                if adx_aligned[i] < 20 or williams_r_aligned[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: trend weakening or overbought condition resolved
                if adx_aligned[i] < 20 or williams_r_aligned[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ADX_WilliamsR_Volume"
timeframe = "6h"
leverage = 1.0