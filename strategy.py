#!/usr/bin/env python3
"""
12h_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict
Strategy: 12h Camarilla pivot R1/S1 breakout with volume confirmation and range filter.
Long: Price breaks above R1 in uptrend with volume spike and non-range market.
Short: Price breaks below S1 in downtrend with volume spike and non-range market.
Designed for 12h timeframe: ~15-25 trades/year per symbol (60-100 total over 4 years).
Works in bull/bear via trend filter and range avoidance.
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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    # Daily EMA50 and EMA200 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily volume average (20-period)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Daily range for range filter (ATR-based)
    atr_14 = pd.Series(high_1d - low_1d).rolling(window=14, min_periods=14).mean().values
    # Range filter: avoid trading when ATR is too low (choppy market)
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    
    # Align all daily data to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    vol_ma_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for EMA200 and ATR MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or
            np.isnan(vol_ma_aligned[i]) or np.isnan(atr_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend conditions
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma_aligned[i]
        
        # Range filter: only trade when volatility is sufficient (not choppy)
        vol_filter = atr_14[i] > 0.5 * atr_ma_aligned[i]  # ATR > 50% of its MA
        
        # Camarilla breakout conditions
        # Long: price breaks above R1
        breakout_long = close[i] > r1_aligned[i]
        # Short: price breaks below S1
        breakout_short = close[i] < s1_aligned[i]
        
        if position == 0:
            # Long: uptrend + volume + volatility filter + breakout above R1
            if uptrend and vol_confirm and vol_filter and breakout_long:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + volume + volatility filter + breakout below S1
            elif downtrend and vol_confirm and vol_filter and breakout_short:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend change, volatility collapse, or break below S1 (reversal)
            if not uptrend or not vol_filter or close[i] < s1_aligned[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend change, volatility collapse, or break above R1 (reversal)
            if not downtrend or not vol_filter or close[i] > r1_aligned[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_R1_S1_Breakout_Volume_RangeFilter_Strict"
timeframe = "12h"
leverage = 1.0