#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 12h ADX regime filter
# - Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure
# - 12h ADX > 25 indicates trending regime; < 20 indicates ranging
# - In trending regime (ADX > 25): go long when Bull Power > 0 and rising, short when Bear Power > 0 and rising
# - In ranging regime (ADX < 20): fade extremes - long when Bear Power < 0 and rising, short when Bull Power < 0 and rising
# - Volume confirmation: require volume > 1.5 * 20-period average to avoid low-vol false signals
# - Target: 12-25 trades/year on 6h (50-100 total over 4 years) to minimize fee drag
# - Works in bull markets via trend following, in bear via mean reversion in ranges

name = "6h_12h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 12h ADX(14) for regime detection
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_12h[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_12h - np.roll(high_12h, 1)) > (np.roll(low_12h, 1) - low_12h), 
                       np.maximum(high_12h - np.roll(high_12h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_12h, 1) - low_12h) > (high_12h - np.roll(high_12h, 1)), 
                        np.maximum(np.roll(low_12h, 1) - low_12h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr_12h).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    dx[np.isnan(dx)] = 0
    
    # ADX
    adx_12h = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # 6h EMA13 for Elder Ray
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Buying pressure
    bear_power = ema13 - low   # Selling pressure
    
    # Momentum (rate of change) of Elder Ray
    bull_power_mom = bull_power - np.roll(bull_power, 6)  # 6-bar momentum
    bear_power_mom = bear_power - np.roll(bear_power, 6)
    bull_power_mom[0:6] = 0
    bear_power_mom[0:6] = 0
    
    # Volume confirmation: > 1.5 * 20-period average
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(bull_power_mom[i]) or np.isnan(bear_power_mom[i]) or
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx_aligned[i]
        bull = bull_power[i]
        bear = bear_power[i]
        bull_mom = bull_power_mom[i]
        bear_mom = bear_power_mom[i]
        vol_ok = volume_confirm[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if adx_val > 25:  # Trending regime - exit when bull power fades
                if bull <= 0 or bull_mom < 0:
                    position = 0
                    signals[i] = 0.0
            elif adx_val < 20:  # Ranging regime - exit when bear power rises (mean reversion)
                if bear >= 0 or bear_mom > 0:
                    position = 0
                    signals[i] = 0.0
            else:  # Transition regime - hold unless clear reversal
                if bull <= 0 and bear >= 0:  # Neutral zone
                    position = 0
                    signals[i] = 0.0
            if position == 1:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            if adx_val > 25:  # Trending regime - exit when bear power fades
                if bear <= 0 or bear_mom < 0:
                    position = 0
                    signals[i] = 0.0
            elif adx_val < 20:  # Ranging regime - exit when bull power rises (mean reversion)
                if bull >= 0 or bull_mom > 0:
                    position = 0
                    signals[i] = 0.0
            else:  # Transition regime - hold unless clear reversal
                if bull <= 0 and bear >= 0:  # Neutral zone
                    position = 0
                    signals[i] = 0.0
            if position == -1:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on regime
            if vol_ok:  # Only trade with volume confirmation
                if adx_val > 25:  # Strong trend - follow Elder Ray momentum
                    if bull > 0 and bull_mom > 0:  # Rising bull power
                        position = 1
                        signals[i] = 0.25
                    elif bear > 0 and bear_mom > 0:  # Rising bear power
                        position = -1
                        signals[i] = -0.25
                elif adx_val < 20:  # Range - mean reversion at extremes
                    if bear < 0 and bear_mom > 0:  # Bear power rising from negative (oversold)
                        position = 1
                        signals[i] = 0.25
                    elif bull < 0 and bull_mom > 0:  # Bull power rising from negative (oversold short)
                        position = -1
                        signals[i] = -0.25
                # Transition regime (20 <= ADX <= 25) - no new entries to avoid whipsaw
    
    return signals