#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_Regime_v1
Hypothesis: On 6h timeframe, trade Elder Ray Bull Power and Bear Power signals filtered by 12h ADX regime and volume confirmation. 
Bull Power = High - EMA(13), Bear Power = EMA(13) - Low. 
In trending markets (ADX > 25): take longs when Bull Power > 0 and rising, shorts when Bear Power > 0 and rising. 
In ranging markets (ADX < 20): fade extremes - short when Bull Power > 0.7*ATR, long when Bear Power > 0.7*ATR. 
Volume spike confirms participation. Designed for 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).
Works in bull/bear/range markets via adaptive regime filtering.
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
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ATR(14) for 6h (used in signals and stops)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr1 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr1])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate EMA(13) for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA
    bear_power = ema13 - low   # Bear Power = EMA - Low
    
    # Rising Bull/Bear Power (1-bar momentum)
    bull_power_rising = bull_power > np.roll(bull_power, 1)
    bear_power_rising = bear_power > np.roll(bear_power, 1)
    # Handle first bar
    bull_power_rising[0] = False
    bear_power_rising[0] = False
    
    # Volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Get 12h data for ADX regime filter
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    # True Range
    tr_12h1 = np.maximum(high_12h[1:] - low_12h[1:], np.abs(high_12h[1:] - close_12h[:-1]))
    tr_12h1 = np.maximum(tr_12h1, np.abs(low_12h[1:] - close_12h[:-1]))
    tr_12h = np.concatenate([[np.nan], tr_12h1])
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]), 
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]), 
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed TR, DM+ , DM-
    tr_14 = pd.Series(tr_12h).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power)  # Wait, wrong - need to align 6h bull_power
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power)   # Wrong again
    
    # Correct: align the 12h ADX to 6h, but keep 6h indicators as-is
    bull_power_aligned = bull_power  # Already 6h
    bear_power_aligned = bear_power  # Already 6h
    bull_power_rising_aligned = bull_power_rising  # Already 6h
    bear_power_rising_aligned = bear_power_rising  # Already 6h
    volume_spike_aligned = volume_spike  # Already 6h
    atr_aligned = atr  # Already 6h
    
    # Only ADX needs alignment from 12h to 6h
    adx_aligned = align_htf_to_ltf(prices, df_12h, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of EMA(13), ATR(14), volume MA(20), ADX periods
    start_idx = max(13, 14, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or
            np.isnan(bear_power_aligned[i]) or
            np.isnan(atr_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        adx_val = adx_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        bull_power_rising_val = bull_power_rising_aligned[i]
        bear_power_rising_val = bear_power_rising_aligned[i]
        vol_spike = volume_spike_aligned[i]
        atr_val = atr_aligned[i]
        close_val = close[i]
        
        if position == 0:
            # Determine regime: trending (ADX > 25) or ranging (ADX < 20)
            if adx_val > 25:
                # Trending regime: trade with momentum
                # Long: Bull Power > 0 and rising, volume spike
                long_signal = (bull_power_val > 0) and bull_power_rising_val and vol_spike
                # Short: Bear Power > 0 and rising, volume spike
                short_signal = (bear_power_val > 0) and bear_power_rising_val and vol_spike
            elif adx_val < 20:
                # Ranging regime: fade extremes
                # Short when Bull Power is excessively high (overbought)
                short_signal = (bull_power_val > 0.7 * atr_val) and vol_spike
                # Long when Bear Power is excessively high (oversold)
                long_signal = (bear_power_val > 0.7 * atr_val) and vol_spike
            else:
                # Transition regime: no new signals
                long_signal = False
                short_signal = False
            
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
            # Exit conditions
            exit_signal = False
            if adx_val > 25:
                # In trend: exit when Bull Power <= 0 or not rising
                exit_signal = (bull_power_val <= 0) or not bull_power_rising_val
            elif adx_val < 20:
                # In range: exit when Bull Power normalizes (< 0.3*ATR)
                exit_signal = bull_power_val < 0.3 * atr_val
            else:
                # Transition: exit when power fades
                exit_signal = bull_power_val < 0.1 * atr_val
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            exit_signal = False
            if adx_val > 25:
                # In trend: exit when Bear Power <= 0 or not rising
                exit_signal = (bear_power_val <= 0) or not bear_power_rising_val
            elif adx_val < 20:
                # In range: exit when Bear Power normalizes (< 0.3*ATR)
                exit_signal = bear_power_val < 0.3 * atr_val
            else:
                # Transition: exit when power fades
                exit_signal = bear_power_val < 0.1 * atr_val
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_Regime_v1"
timeframe = "6h"
leverage = 1.0