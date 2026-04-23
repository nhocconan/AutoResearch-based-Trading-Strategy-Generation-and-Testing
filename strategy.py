#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray regime and volume spike confirmation.
- Williams %R(14) < -80 = oversold (long setup), > -20 = overbought (short setup)
- Elder Ray: Bull Power = High - EMA13(close), Bear Power = Low - EMA13(close)
  Bull Power > 0 AND rising = bullish regime, Bear Power < 0 AND falling = bearish regime
- Volume: Current 6h volume > 1.5x 20-period MA for confirmation
- Only trade in direction of Elder Ray regime: long when bullish, short when bearish
- Exit when Williams %R reverses to opposite extreme or regime changes
Uses 1d HTF for Elder Ray regime to avoid counter-trend trades, Williams %R for mean reversion entries,
volume spike to ensure momentum. Designed for 6h timeframe to capture swing reversals in both bull and bear markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R (14-period)
    lookback_willr = 14
    willr = np.full(n, np.nan)
    
    for i in range(lookback_willr - 1, n):
        highest_high = np.max(high[i-lookback_willr+1:i+1])
        lowest_low = np.min(low[i-lookback_willr+1:i+1])
        if highest_high != lowest_low:
            willr[i] = (highest_high - close[i]) / (highest_high - lowest_low) * -100
        else:
            willr[i] = -50  # Avoid division by zero
    
    # Calculate 1d Elder Ray for regime filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power_1d = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    # Align Elder Ray components to LTF
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback_willr - 1, 13, 20)  # Williams %R, EMA13, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(willr[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        willr_val = willr[i]
        bull_power = bull_power_aligned[i]
        bear_power = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Elder Ray slopes for regime direction
        if i >= start_idx + 1:
            bull_power_prev = bull_power_aligned[i-1]
            bear_power_prev = bear_power_aligned[i-1]
            bull_regime = bull_power > 0 and bull_power > bull_power_prev  # Bull Power > 0 AND rising
            bear_regime = bear_power < 0 and bear_power < bear_power_prev  # Bear Power < 0 AND falling
        else:
            bull_regime = False
            bear_regime = False
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R oversold (< -80) AND bullish regime AND volume filter
            if willr_val < -80 and bull_regime and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) AND bearish regime AND volume filter
            elif willr_val > -20 and bear_regime and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R reverses above -50 OR regime turns bearish
                if willr_val > -50 or not bull_regime:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R reverses below -50 OR regime turns bullish
                if willr_val < -50 or not bear_regime:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_Regime_VolumeSpike"
timeframe = "6h"
leverage = 1.0