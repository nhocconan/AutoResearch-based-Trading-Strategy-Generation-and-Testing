# 4h Camarilla Pivot Reversal with Volume Spike and ADX Trend Filter
# Target: 20-40 trades/year on 4H (80-160 total over 4 years)
# Uses daily Camarilla pivot levels (S1/R1, S2/R2) for mean reversion in ranging markets
# Filters by ADX < 25 (ranging) and volume > 1.5x 20-period average for confirmation
# Works in both bull/bear markets by fading extremes during low volatility regimes

from typing import Any
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_PivotReversal_Volume_ADX"
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
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    r2 = np.zeros_like(close_1d)
    s2 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        pivot[i] = (ph + pl + pc) / 3.0
        range_ = ph - pl
        
        # Camarilla levels
        r1[i] = pc + (range_ * 1.1 / 12)
        s1[i] = pc - (range_ * 1.1 / 12)
        r2[i] = pc + (range_ * 1.1 / 6)
        s2[i] = pc - (range_ * 1.1 / 6)
    
    # First day has no prior data
    pivot[0] = r1[0] = s1[0] = r2[0] = s2[0] = np.nan
    
    # Align Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # ADX for trend/ranging filter (using 4h data)
    # Calculate +DM, -DM, TR
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    
    up_move = high_series.diff()
    down_move = low_series.diff() * -1
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift())
    tr3 = abs(low_series - close_series.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Smoothed values
    period = 14
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().values
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values
    
    # Range filter: ADX < 25 indicates ranging market
    ranging = adx < 25
    
    # Volume confirmation: current volume > 1.5x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(adx[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: price at S1 support in ranging market with volume
            if (ranging[i] and
                close[i] <= s1_aligned[i] * 1.002 and  # Within 0.2% of S1
                close[i] >= s1_aligned[i] * 0.998 and
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: price at R1 resistance in ranging market with volume
            elif (ranging[i] and
                  close[i] >= r1_aligned[i] * 0.998 and  # Within 0.2% of R1
                  close[i] <= r1_aligned[i] * 1.002 and
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches pivot or S2 broken
            if close[i] >= pivot_aligned[i] * 0.998 or close[i] <= s2_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches pivot or R2 broken
            if close[i] <= pivot_aligned[i] * 1.002 or close[i] >= r2_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals