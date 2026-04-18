#!/usr/bin/env python3
"""
12h Camarilla Pivot R1/S1 Breakout with Volume Confirmation and ADX Filter
Hypothesis: Price breaking above/below Camarilla R1/S1 levels calculated from daily candles
with volume confirmation (volume > 1.5x 20-period EMA) and trend strength (ADX > 20)
indicates strong momentum. Camarilla levels from daily timeframe provide robust support/resistance
levels that work in both bull and bear markets. Target: 15-30 trades/year to minimize fee drain.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from daily data
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (wait for daily candle to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # ADX for trend strength (14-period) on 12h data
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 14,14,20)
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1_level = r1_aligned[i]
        s1_level = s1_aligned[i]
        adx_val = adx[i]
        vol_conf = vol_ratio[i] > 1.5
        
        if position == 0:
            # Strong trend (ADX > 20) and volume confirmation
            # Price breaks above R1 = long
            if adx_val > 20 and price > r1_level and vol_conf:
                signals[i] = 0.25
                position = 1
            # Price breaks below S1 = short
            elif adx_val > 20 and price < s1_level and vol_conf:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit if trend weakens or price returns to S1 level (mean reversion)
            if adx_val < 15 or price < s1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if trend weakens or price returns to R1 level (mean reversion)
            if adx_val < 15 or price > r1_level:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_ADX"
timeframe = "12h"
leverage = 1.0