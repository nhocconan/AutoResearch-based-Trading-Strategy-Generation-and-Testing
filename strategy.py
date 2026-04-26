#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wATR_Trend_v1
Hypothesis: Trade 1d Camarilla R1/S1 breakouts filtered by 1w ATR-based trend strength and volume confirmation.
Camarilla pivot levels provide institutional support/resistance. Breakouts with strong weekly trend (ADX>25) 
and volume confirmation have edge in both bull/bear markets. ATR filter avoids choppy markets.
Target: 7-25 trades/year (30-100 total) to minimize fee drag on 1d timeframe.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (R1, S1)
    # Based on previous day's OHLC
    phigh = np.roll(df_1d['high'].values, 1)  # previous day high
    plow = np.roll(df_1d['low'].values, 1)    # previous day low
    pclose = np.roll(df_1d['close'].values, 1) # previous day close
    
    # Camarilla: R1 = c + (h-l)*1.1/12, S1 = c - (h-l)*1.1/12
    camarilla_r1 = pclose + (phigh - plow) * 1.1 / 12
    camarilla_s1 = pclose - (phigh - plow) * 1.1 / 12
    
    # Align Camarilla levels to 1d timeframe (already aligned via get_htf_data)
    # No need to align since we're working at 1d timeframe directly
    
    # Get 1w data for HTF trend filter (ADX-based)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate 1w ADX(14) for trend strength
    whigh = df_1w['high'].values
    wlow = df_1w['low'].values
    wclose = df_1w['close'].values
    
    # True Range
    tr1 = whigh[1:] - wlow[1:]
    tr2 = np.abs(whigh[1:] - wclose[:-1])
    tr3 = np.abs(wlow[1:] - wclose[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # Directional Movement
    dm_plus = np.where((whigh[1:] - whigh[:-1]) > (wlow[:-1] - wlow[1:]), 
                       np.maximum(whigh[1:] - whigh[:-1], 0), 0)
    dm_minus = np.where((wlow[:-1] - wlow[1:]) > (whigh[1:] - whigh[:-1]), 
                        np.maximum(wlow[:-1] - wlow[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed TR, DM+
    tr_period = 14
    atr = pd.Series(tr).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=tr_period, adjust=False, min_periods=tr_period).mean().values
    
    # Volume confirmation: 1.5x median volume on 1d
    vol_median_1d = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough for pivots (1), ADX (34), volume median (20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Current bar OHLC
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        
        # Previous day's Camarilla levels (from 1d data)
        # Since we're at 1d timeframe, use index i-1 for previous day
        if i-1 < 0 or i-1 >= len(camarilla_r1):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        r1_val = camarilla_r1[i-1]
        s1_val = camarilla_s1[i-1]
        
        # Skip if any data not ready
        if (np.isnan(adx[i]) or 
            np.isnan(vol_median_1d[i]) or
            np.isnan(r1_val) or np.isnan(s1_val)):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        adx_val = adx[i]
        vol_median_val = vol_median_1d[i]
        volume_val = volume[i]
        
        if position == 0:
            # Long: break above R1 with strong trend (ADX>25) and volume confirmation
            long_signal = (high_val > r1_val) and \
                          (adx_val > 25) and \
                          (volume_val > 1.5 * vol_median_val)
            # Short: break below S1 with strong trend (ADX>25) and volume confirmation
            short_signal = (low_val < s1_val) and \
                           (adx_val > 25) and \
                           (volume_val > 1.5 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = 0.25
            # Exit: trend weakening (ADX<20) or price returns to Camarilla H3/L3 level
            # For simplicity, exit on trend weakening or opposite touch
            if adx_val < 20 or low_val < s1_val:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            signals[i] = -0.25
            # Exit: trend weakening (ADX<20) or price returns to Camarilla H3/L3 level
            if adx_val < 20 or high_val > r1_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wATR_Trend_v1"
timeframe = "1d"
leverage = 1.0