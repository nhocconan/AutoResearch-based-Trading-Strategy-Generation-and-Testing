#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_1dTrend_HTFVolume_v1
Hypothesis: Weekly pivot levels (from prior week) act as strong support/resistance. Donchian(20) breakout in direction of 1d EMA50 trend with HTF volume confirmation (>1.3x weekly median volume) captures momentum after pivot rejection/breakout. Works in bull/bear by only trading with 1d trend. Targets 12-37 trades/year via tight confluence of weekly pivot, Donchian breakout, trend, and volume.
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
    
    # Get 1d data for HTF trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly pivot from prior week OHLC
    whigh = pd.Series(df_1w['high'].values).shift(1).values
    wlow = pd.Series(df_1w['low'].values).shift(1).values
    wclose = pd.Series(df_1w['close'].values).shift(1).values
    
    # Weekly pivot levels (standard calculation)
    pivot = (whigh + wlow + wclose) / 3.0
    r1 = 2 * pivot - wlow
    s1 = 2 * pivot - whigh
    r2 = pivot + (whigh - wlow)
    s2 = pivot - (whigh - wlow)
    r3 = whigh + 2 * (pivot - wlow)
    s3 = wlow - 2 * (whigh - pivot)
    
    # Donchian(20) channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly median volume for confirmation
    vol_weekly = pd.Series(df_1w['volume'].values)
    vol_median_weekly = vol_weekly.rolling(window=4, min_periods=4).median().values  # ~1 month lookback
    
    # Align HTF indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    donch_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}, index=prices.index), donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}, index=prices.index), donch_low)
    vol_median_weekly_aligned = align_htf_to_ltf(prices, df_1w, vol_median_weekly)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(50) 1d, weekly pivot (need 2 bars), Donchian(20), weekly volume median (4)
    start_idx = max(50, 2, 20, 4) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(donch_high_aligned[i]) or
            np.isnan(donch_low_aligned[i]) or
            np.isnan(vol_median_weekly_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        ema_50_1d_val = ema_50_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        volume_val = volume[i]
        vol_median_weekly_val = vol_median_weekly_aligned[i]
        pivot_val = pivot_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        
        # Trend filter: price > EMA50 (uptrend) or < EMA50 (downtrend)
        uptrend = close_val > ema_50_1d_val
        downtrend = close_val < ema_50_1d_val
        
        # HTF volume confirmation: volume > 1.3x weekly median volume
        htf_volume_spike = volume_val > 1.3 * vol_median_weekly_val
        
        if position == 0:
            # Long: Donchian breakout above weekly R2 with volume spike, and uptrend
            long_signal = (close_val > donch_high_val) and \
                          (donch_high_val > r2_val) and \
                          htf_volume_spike and \
                          uptrend
            
            # Short: Donchian breakout below weekly S2 with volume spike, and downtrend
            short_signal = (close_val < donch_low_val) and \
                           (donch_low_val < s2_val) and \
                           htf_volume_spike and \
                           downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below Donchian low or trend reversal
            if close_val < donch_low_val or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above Donchian high or trend reversal
            if close_val > donch_high_val or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_1dTrend_HTFVolume_v1"
timeframe = "6h"
leverage = 1.0