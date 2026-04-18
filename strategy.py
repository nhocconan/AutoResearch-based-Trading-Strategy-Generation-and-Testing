#!/usr/bin/env python3
"""
4h_Pivot_R1S1_Breakout_Volume_ATRFilter_v2
Breakout strategy using Camarilla Pivot points with volume confirmation and ATR filter.
- Long when price breaks above R1 with volume spike > 1.5x average and ATR < 20-day percentile
- Short when price breaks below S1 with volume spike > 1.5x average and ATR < 20-day percentile
- Exit when price returns to pivot point (PP) or ATR exceeds threshold
- Uses 12h timeframe for trend filter (EMA34) to avoid counter-trend trades
- Designed for 20-30 trades/year per symbol with controlled risk
Works in bull markets (breakouts) and bear markets (breakdowns) with volatility filter
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla_pivot(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close."""
    range_val = high - low
    pp = (high + low + close) / 3
    r1 = close + range_val * 1.1 / 12
    s1 = close - range_val * 1.1 / 12
    r2 = close + range_val * 1.1 / 6
    s2 = close - range_val * 1.1 / 6
    r3 = close + range_val * 1.1 / 4
    s3 = close - range_val * 1.1 / 4
    r4 = close + range_val * 1.1 / 2
    s4 = close - range_val * 1.1 / 2
    return pp, r1, s1, r2, s2, r3, s3, r4, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34 for trend filter
    close_12h_series = pd.Series(close_12h)
    ema_34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate ATR(20) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 20-period ATR percentile (20-day lookback)
    atr_percentile = pd.Series(atr).rolling(window=20, min_periods=1).rank(pct=True).values
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(atr_percentile[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        # Calculate Camarilla pivot for previous period
        pp, r1, s1, r2, s2, r3, s3, r4, s4 = calculate_camarilla_pivot(
            high[i-1], low[i-1], close[i-1]
        )
        
        # Volume spike condition (> 1.5x average)
        volume_spike = volume[i] > vol_ma[i] * 1.5
        
        # ATR filter: low volatility environment (ATR below 30th percentile)
        atr_filter = atr_percentile[i] < 0.3
        
        # 12h trend filter: only take longs in uptrend, shorts in downtrend
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume spike, low vol, and uptrend
            if close[i] > r1 and volume_spike and atr_filter and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike, low vol, and downtrend
            elif close[i] < s1 and volume_spike and atr_filter and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price returns to pivot point or volatility increases
            if close[i] < pp or atr_percentile[i] > 0.7:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or volatility increases
            if close[i] > pp or atr_percentile[i] > 0.7:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Pivot_R1S1_Breakout_Volume_ATRFilter_v2"
timeframe = "4h"
leverage = 1.0