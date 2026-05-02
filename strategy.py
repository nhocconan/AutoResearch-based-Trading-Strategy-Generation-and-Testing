#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray Power combination with 1d regime filter
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) provides trend direction and filter
# Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures trend strength
# 1d Chopiness Index regime filter: only trade when CHOP > 61.8 (ranging) for mean reversion
# Uses Camarilla R3/S3 levels from 6h for precise entry/exit in ranging markets
# Volume confirmation (1.5x 20-period average) ensures participation
# Session filter (08-20 UTC) reduces noise
# Designed for low frequency: targets 80-120 total trades over 4 years = 20-30/year
# Works in ranging markets via mean reversion at extremes, avoids trends via Alligator alignment filter

name = "6h_Alligator_ElderRay_Camarilla_MR_1dChop_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 1d data ONCE before loop for Chop regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Chopiness Index (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR14
    atr1 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Chop = 100 * log15(sum(ATR14)/ (max(high)-min(low)) over 14 periods)
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    denominator = max_high - min_low
    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)
    chop = 100 * np.log15(atr1 * 14 / denominator)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Williams Alligator on 6h timeframe (using input prices directly)
    # Jaw: 13-period SMMA (smoothed) of median price
    # Teeth: 8-period SMMA of median price
    # Lips: 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    # SMMA calculation (similar to Wilder's smoothing)
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_value) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Elder Ray Power (using EMA13 as approximate to SMMA13 for power calculation)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = ema_13 - low   # Bear Power = EMA13 - Low
    
    # Calculate Camarilla pivot levels from previous 6h bar
    typical_price = (high + low + close) / 3.0
    rng = high - low
    camarilla_h4 = typical_price + 1.1 * rng / 2.0  # R3
    camarilla_l4 = typical_price - 1.1 * rng / 2.0  # S3
    camarilla_h2 = typical_price + 1.1 * rng / 6.0  # R1
    camarilla_l2 = typical_price - 1.1 * rng / 6.0  # S1
    
    # Shift to align with bar close (use previous bar's levels)
    camarilla_h4 = np.roll(camarilla_h4, 1)
    camarilla_l4 = np.roll(camarilla_l4, 1)
    camarilla_h2 = np.roll(camarilla_h2, 1)
    camarilla_l2 = np.roll(camarilla_l2, 1)
    camarilla_h4[0] = np.nan
    camarilla_l4[0] = np.nan
    camarilla_h2[0] = np.nan
    camarilla_l2[0] = np.nan
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(camarilla_h4[i]) or
            np.isnan(camarilla_l4[i]) or np.isnan(camarilla_h2[i]) or
            np.isnan(camarilla_l2[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: only trade in ranging markets (Chop > 61.8)
        if chop_aligned[i] <= 61.8:
            signals[i] = 0.0
            continue
        
        # Alligator alignment check: Jaw > Teeth > Lips = downtrend, Lips > Teeth > Jaw = uptrend
        # For mean reversion, we want weak alignment (not strongly trending)
        alligator_aligned_up = lips[i] > teeth[i] > jaw[i]
        alligator_aligned_down = jaw[i] > teeth[i] > lips[i]
        weak_alligator = not (alligator_aligned_up or alligator_aligned_down)
        
        if position == 0:  # Flat - look for new entries
            # Long: Price at Camarilla S3 (support) + weak bear power + volume confirm
            if close[i] <= camarilla_l4[i] and bear_power[i] < 0 and volume_confirm[i] and weak_alligator:
                signals[i] = 0.25
                position = 1
            # Short: Price at Camarilla R3 (resistance) + weak bull power + volume confirm
            elif close[i] >= camarilla_h4[i] and bull_power[i] > 0 and volume_confirm[i] and weak_alligator:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price reaches Camarilla R1 or strong bull power develops
            if close[i] >= camarilla_h2[i] or bull_power[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price reaches Camarilla S1 or strong bear power develops
            if close[i] <= camarilla_l2[i] or bear_power[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals