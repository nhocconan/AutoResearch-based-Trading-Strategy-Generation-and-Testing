#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_Breakout_With_Trend_Filter
Hypothesis:
- Camarilla pivot levels (R3/S3, R4/S4) on daily timeframe act as strong support/resistance.
- Breakout beyond R4/S4 with continuation (close beyond level) indicates institutional interest.
- Weekly trend filter (EMA200) ensures we only take breakouts in direction of higher timeframe trend.
- Volume confirmation filters false breakouts.
- Works in bull/bear: trend filter avoids counter-trend breakouts; Camarilla levels adapt to volatility.
- Target: 15-30 trades/year per symbol (60-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Camarilla_Breakout_With_Trend_Filter"
timeframe = "6h"
leverage = 1.0

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels for given high, low, close.
    Returns: (S3, S2, S1, PP, R1, R2, R3, R4)
    Formula:
    PP = (H + L + C) / 3
    R4 = C + ((H-L) * 1.1/2)
    R3 = C + ((H-L) * 1.1/4)
    R2 = C + ((H-L) * 1.1/6)
    R1 = C + ((H-L) * 1.1/12)
    S1 = C - ((H-L) * 1.1/12)
    S2 = C - ((H-L) * 1.1/6)
    S3 = C - ((H-L) * 1.1/4)
    S4 = C - ((H-L) * 1.1/2)
    """
    typical_range = high - low
    pp = (high + low + close) / 3.0
    r4 = close + typical_range * 1.1 / 2.0
    r3 = close + typical_range * 1.1 / 4.0
    r2 = close + typical_range * 1.1 / 6.0
    r1 = close + typical_range * 1.1 / 12.0
    s1 = close - typical_range * 1.1 / 12.0
    s2 = close - typical_range * 1.1 / 6.0
    s3 = close - typical_range * 1.1 / 4.0
    s4 = close - typical_range * 1.1 / 2.0
    return s3, s2, s1, pp, r1, r2, r3, r4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # === Get daily data for Camarilla levels (once before loop) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla for each daily bar using prior day's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's levels (avoid look-ahead)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    camarilla_r4 = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        s3, s2, s1, pp, r1, r2, r3, r4 = calculate_camarilla(
            high_1d[i-1], low_1d[i-1], close_1d[i-1]
        )
        camarilla_s3[i] = s3
        camarilla_r4[i] = r4
    
    # Align to 6h timeframe
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    
    # === Weekly trend filter: EMA200 on weekly close ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 6h price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume filter: current vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        s3_val = camarilla_s3_aligned[i]
        r4_val = camarilla_r4_aligned[i]
        ema_weekly = ema200_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(s3_val) or np.isnan(r4_val) or 
            np.isnan(ema_weekly) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price closes above R4 in uptrend with volume
            if (close_val > r4_val and 
                close_val > ema_weekly and 
                vol_ratio_val > 1.5):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price closes below S3 in downtrend with volume
            elif (close_val < s3_val and 
                  close_val < ema_weekly and 
                  vol_ratio_val > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price breaks below S3 (mean reversion) or trend change
            if close_val < s3_val or close_val < ema_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price breaks above R4 (mean reversion) or trend change
            if close_val > r4_val or close_val > ema_weekly:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals