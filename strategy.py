#!/usr/bin/env python3
"""
Hypothesis: 1-day Bollinger Band squeeze breakout with 1-week volume confirmation and ADX trend filter.
Long when price breaks above upper BB during low volatility (BBW < 50th percentile) and weekly volume > 1.5x average.
Short when price breaks below lower BB under same conditions.
Exit when price returns to middle BB or volatility expands (BBW > 80th percentile).
Designed for low-frequency, high-conviction trades (~10-25/year) to capture explosive moves after consolidation.
Works in bull markets (breakouts continuation) and bear markets (breakdowns continuation).
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
    
    # Calculate 1-day Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper = basis + 2 * dev
    lower = basis - 2 * dev
    
    # Bollinger Band Width for squeeze detection
    bbw = (upper - lower) / basis
    bbw_percentile = pd.Series(bbw).rolling(window=50, min_periods=50).rank(pct=True) * 100
    
    # Weekly volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    weekly_volume = df_1w['volume'].values
    weekly_volume_ma = pd.Series(weekly_volume).rolling(window=10, min_periods=10).mean()
    weekly_volume_ratio = weekly_volume / weekly_volume_ma
    weekly_volume_ratio_aligned = align_htf_to_ltf(prices, df_1w, weekly_volume_ratio.values)
    
    # ADX filter for trend strength (weekly)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum()
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum()
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum()
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean()
    adx_values = adx.values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bbw_percentile.iloc[i]) or np.isnan(upper.iloc[i]) or 
            np.isnan(lower.iloc[i]) or np.isnan(basis.iloc[i]) or
            np.isnan(weekly_volume_ratio_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        bbw_p = bbw_percentile.iloc[i]
        vol_ratio = weekly_volume_ratio_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: BB squeeze (low volatility) + weekly volume surge + ADX strength + breakout above upper BB
            if bbw_p < 50 and vol_ratio > 1.5 and adx_val > 20 and close[i] > upper.iloc[i]:
                signals[i] = 0.25
                position = 1
            # Short: BB squeeze + weekly volume surge + ADX strength + breakdown below lower BB
            elif bbw_p < 50 and vol_ratio > 1.5 and adx_val > 20 and close[i] < lower.iloc[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: return to middle BB or volatility expansion (high BBW)
                if close[i] < basis.iloc[i] or bbw_p > 80:
                    exit_signal = True
            else:  # position == -1
                # Exit short: return to middle BB or volatility expansion
                if close[i] > basis.iloc[i] or bbw_p > 80:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_BB_Squeeze_Volume_ADX_Breakout"
timeframe = "1d"
leverage = 1.0