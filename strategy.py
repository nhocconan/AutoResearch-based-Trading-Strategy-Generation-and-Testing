#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level breakout with volume confirmation and 1d trend filter
# - Uses Camarilla pivot levels (S1, S2, R1, R2) from previous 1d for breakout entries
# - Long when price breaks above R1 with volume > 1.5x 20-period average
# - Short when price breaks below S1 with volume > 1.5x 20-period average
# - Trend filter: only take longs when price > 1d EMA50, shorts when price < 1d EMA50
# - Exit when price returns to the 1d pivot point (central pivot)
# - Position size: 0.25 (25%) to balance profit potential and drawdown control
# - Designed to work in trending markets by following higher timeframe trend
# - Target: ~25-40 trades/year to minimize fee drag while capturing meaningful moves

name = "4h_Camarilla_Pivot_Breakout_Volume_TrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous 1d OHLC
    # Formulas: 
    # Pivot = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1/12
    # S1 = C - (H - L) * 1.1/12
    # R2 = C + (H - L) * 1.1/6
    # S2 = C - (H - L) * 1.1/6
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    range_hl = df_1d['high'] - df_1d['low']
    
    pivot = typical_price.values
    r1 = df_1d['close'].values + range_hl.values * 1.1 / 12
    s1 = df_1d['close'].values - range_hl.values * 1.1 / 12
    r2 = df_1d['close'].values + range_hl.values * 1.1 / 6
    s2 = df_1d['close'].values - range_hl.values * 1.1 / 6
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume average (20-period) for volume confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Pre-compute session filter (08:00-20:00 UTC) - optional but helps avoid low-volume periods
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session (optional)
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Look for long entry: price breaks above R1 + uptrend (price > EMA50) + volume
            if close[i] > r1_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below S1 + downtrend (price < EMA50) + volume
            elif close[i] < s1_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price returns to pivot point (mean reversion to fair value)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price returns to pivot point (mean reversion to fair value)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals