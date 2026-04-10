#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d ADX regime filter
# - Primary signal: Elder Ray Bull Power (high - EMA13) and Bear Power (low - EMA13) from 6h
# - Regime filter: 1d ADX > 25 (trending market) - only take signals in trending conditions
# - Entry: Long when Bull Power > 0 and previous Bear Power < 0 (bullish reversal)
#          Short when Bear Power < 0 and previous Bull Power > 0 (bearish reversal)
# - Exit: Opposite Elder Ray signal or ATR-based stoploss (2.0x ATR)
# - Position size: 0.25 discrete level to minimize fee churn
# - Target: 50-150 total trades over 4 years (12-37/year) per 6h strategy guidelines

name = "6h_1d_elderray_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Pre-compute 6h EMA(13) for Elder Ray
    close_6h = prices['close'].values
    ema_13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    bull_power = high_6h - ema_13  # Bull Power = High - EMA13
    bear_power = low_6h - ema_13   # Bear Power = Low - EMA13
    
    # Pre-compute 6h ATR(20) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bear Power becomes positive OR stoploss hit
            if bear_power[i] > 0 or close_6h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power becomes negative OR stoploss hit
            if bull_power[i] < 0 or close_6h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Elder Ray reversals in trending market (ADX > 25)
            if adx_aligned[i] > 25.0:
                # Long: Bull Power turns positive from negative (bullish reversal)
                if bull_power[i] > 0 and i > 100 and bull_power[i-1] <= 0:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: Bear Power turns negative from positive (bearish reversal)
                elif bear_power[i] < 0 and i > 100 and bear_power[i-1] >= 0:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals