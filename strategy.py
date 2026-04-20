#!/usr/bin/env python3
"""
1h_4d1d_Camarilla_Pivot_R1S1_Breakout_VolumeTrend_v1
Concept: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filter.
- Uses daily Camarilla pivot points (R1, S1) for key levels
- Long when price breaks above R1 with volume > 1.5x avg and above 4h EMA20
- Short when price breaks below S1 with volume > 1.5x avg and below 4h EMA20
- Exit when price returns to central pivot (mean reversion)
- Session filter: 08-20 UTC to avoid low-volume hours
- Conservative sizing (0.20) to manage drawdown
- Works in bull/bear: Pivot points adapt, volume confirms, trend filter avoids counter-trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d1d_Camarilla_Pivot_R1S1_Breakout_VolumeTrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # === Calculate daily Camarilla pivot points ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_ = high_1d - low_1d
    r1 = close_1d + (range_ * 1.1 / 12)
    s1 = close_1d - (range_ * 1.1 / 12)
    
    # Align pivot levels to 1h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 4h: EMA20 trend filter ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # === 1h: Volume ratio (current vs 20-period average) ===
    close = prices['close'].values
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume MA20
    
    for i in range(start_idx, n):
        # Skip if outside session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        ema20_4h_val = ema20_4h_aligned[i]
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema20_4h_val) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and above 4h EMA20
            breakout_long = close_val > r1_val
            vol_confirm = vol_ratio_val > 1.5  # Volume above 1.5x average
            
            if breakout_long and vol_confirm and close_val > ema20_4h_val:
                signals[i] = 0.20
                position = 1
            # Short: Price breaks below S1 with volume confirmation and below 4h EMA20
            elif close_val < s1_val and vol_confirm and close_val < ema20_4h_val:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price returns to or below central pivot (mean reversion)
            if close_val <= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price returns to or above central pivot (mean reversion)
            if close_val >= pivot_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals