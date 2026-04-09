#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Williams %R from 1d with regime filter
# - Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13) on 1d
# - Williams %R(14) on 1d for overbought/oversold
# - Regime: ADX(14) on 1d > 25 for trending, < 20 for ranging
# - In trending regime (ADX>25): Long when Bull Power > 0 and %R < -50, Short when Bear Power < 0 and %R > -50
# - In ranging regime (ADX<20): Long when Bull Power < 0 and %R < -80, Short when Bear Power > 0 and %R > -20
# - Position size: 0.25 (25% of capital) for balanced risk/return
# - Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
# - Works in bull markets (trend following) and bear markets (mean reversion in ranges)
# - Combines momentum (Elder Ray) with reversal signals (%R extremes) adapted to regime

name = "6h_1d_elderray_williamsr_regime_v1"
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
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d EMA(13) for Elder Ray
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # 1d Elder Ray components
    bull_power = high_1d - ema_13
    bear_power = low_1d - ema_13
    
    # 1d Williams %R(14)
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # 1d ADX(14) for regime detection
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    
    # Smoothed TR, DM+, DM-
    tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx = np.where((di_plus + di_minus) == 0, 0, dx)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(williams_r_aligned[i]) or np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        bull = bull_power_aligned[i]
        bear = bear_power_aligned[i]
        wr = williams_r_aligned[i]
        adx_val = adx_aligned[i]
        
        if position == 0:  # Flat - look for entry
            # Trending regime (ADX > 25)
            if adx_val > 25:
                # Long: Bull Power positive and Williams %R below -50 (not overbought)
                if bull > 0 and wr < -50:
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power negative and Williams %R above -50 (not oversold)
                elif bear < 0 and wr > -50:
                    position = -1
                    signals[i] = -0.25
            # Ranging regime (ADX < 20)
            elif adx_val < 20:
                # Long: Bull Power negative (weakness) and Williams %R deeply oversold
                if bull < 0 and wr < -80:
                    position = 1
                    signals[i] = 0.25
                # Short: Bear Power positive (strength) and Williams %R deeply overbought
                elif bear > 0 and wr > -20:
                    position = -1
                    signals[i] = -0.25
        
        elif position == 1:  # Long position - look for exit
            # Exit conditions: reverse signal or extreme reversal
            if (bull < 0 and wr > -20) or (bear > 0 and wr < -80):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position - look for exit
            # Exit conditions: reverse signal or extreme reversal
            if (bull > 0 and wr < -80) or (bear < 0 and wr > -20):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals