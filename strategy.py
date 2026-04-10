#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) measures bull/bear strength
# - Primary signal: Bull Power > 0 AND Bear Power < 0 with rising Bull Power = long
#                   Bear Power < 0 AND Bull Power > 0 with falling Bear Power = short
# - Regime filter: 12h ADX > 25 (trending market) enables trend following
# - In ranging markets (ADX < 20): fade extreme Elder Ray readings (mean reversion)
# - Works in bull/bear: ADX adapts to market regime, Elder Ray shows momentum shifts
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines

name = "6h_12h_elderray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Pre-compute 12h ADX(14)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h),
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)),
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(tr_14 > 0, 100 * dm_plus_14 / tr_14, 0)
    di_minus = np.where(tr_14 > 0, 100 * dm_minus_14 / tr_14, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    # Pre-compute 6h EMA13 for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13  # High - EMA13
    bear_power = low_6h - ema_13   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power turns positive OR ADX weakens (range)
            if bear_power[i] > 0 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power turns negative OR ADX weakens (range)
            if bull_power[i] < 0 or adx_aligned[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray signals with ADX regime filter
            if adx_aligned[i] > 25:  # trending market - trend following
                # Long: Bull Power > 0 AND Bear Power < 0 AND Bull Power rising
                if (bull_power[i] > 0 and bear_power[i] < 0 and 
                    i > 100 and bull_power[i] > bull_power[i-1]):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bear Power < 0 AND Bull Power > 0 AND Bear Power falling
                elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                      i > 100 and bear_power[i] < bear_power[i-1]):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
            else:  # ranging market (ADX < 20) - mean reversion
                # Long: Bull Power at extreme negative (oversold)
                if bull_power[i] < -0.5 * np.std(bull_power[max(0, i-50):i+1]):
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bull Power at extreme positive (overbought)
                elif bull_power[i] > 0.5 * np.std(bull_power[max(0, i-50):i+1]):
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals