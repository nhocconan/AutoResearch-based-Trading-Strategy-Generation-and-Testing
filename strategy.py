# 12h_Camarilla_R1S1_Breakout_Volume_Trend_v1
# Hypothesis: Camarilla pivot R1/S1 breakouts on 12h timeframe with volume confirmation and trend filter.
# Uses 1-day Camarilla levels for structure, volume spike for momentum confirmation, and 1-week EMA200 for trend filter.
# Works in bull/bear: Trend filter avoids counter-trend trades, volume confirms breakout strength.
# Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === Daily: Calculate Camarilla pivot levels (using previous day's OHLC) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point = (H + L + C) / 3
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H - L) * 1.1/12
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    # S1 = C - (H - L) * 1.1/12
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align Camarilla levels to 12h timeframe (wait for daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === Weekly: EMA200 trend filter ===
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # === 12h: Price and volume data ===
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Get values
        close_val = close[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema200_val = ema200_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(pivot_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(ema200_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 with volume confirmation and above weekly EMA200
            breakout_long = close_val > r1_val
            trend_filter = close_val > ema200_val
            vol_confirm = vol_ratio_val > 1.8  # Volume spike
            
            if breakout_long and trend_filter and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume confirmation and below weekly EMA200
            elif close_val < s1_val and close_val < ema200_val and vol_ratio_val > 1.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price returns below pivot (mean reversion) or trend fails
            if close_val < pivot_val or close_val < ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price returns above pivot (mean reversion) or trend fails
            if close_val > pivot_val or close_val > ema200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals