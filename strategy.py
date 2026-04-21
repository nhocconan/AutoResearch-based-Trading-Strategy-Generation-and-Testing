#!/usr/bin/env python3
"""
6h_ADX_Alligator_ElderRay_Combo
Hypothesis: Combine ADX trend strength (>25) with Williams Alligator (jaw/teeth/lips alignment) and Elder Ray (bull/bear power) to filter false breakouts.
Alligator confirms trend direction, ADX confirms trend strength, Elder Ray provides entry timing on pullbacks.
Works in both bull and bear markets by requiring alignment across all three indicators.
Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for higher timeframe context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Williams Alligator (jaw/teeth/lips) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Alligator lines: SMAs of median price with different periods
    median_1d = (high_1d + low_1d) / 2.0
    jaw = pd.Series(median_1d).rolling(window=13, min_periods=13).mean().shift(8).values  # blue line
    teeth = pd.Series(median_1d).rolling(window=8, min_periods=8).mean().shift(5).values   # red line
    lips = pd.Series(median_1d).rolling(window=5, min_periods=5).mean().shift(3).values    # green line
    
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # === 1d ADX for trend strength ===
    # Calculate +DM, -DM, TR
    high_diff = np.diff(high_1d, prepend=high_1d[0])
    low_diff = -np.diff(low_1d, prepend=low_1d[0])
    
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Smoothed values (Wilder's smoothing)
    atr_period = 14
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values / atr
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 6h Elder Ray (Bull/Bear Power) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # 13-period EMA for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Smooth the power signals
    bull_power_smooth = pd.Series(bull_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    bear_power_smooth = pd.Series(bear_power).ewm(span=8, adjust=False, min_periods=8).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(30, n):
        # Skip if indicators not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Alligator alignment: lips > teeth > jaw = bullish, lips < teeth < jaw = bearish
        bullish_alligator = (lips_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > jaw_aligned[i])
        bearish_alligator = (lips_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < jaw_aligned[i])
        
        # ADX trend strength filter
        strong_trend = adx_aligned[i] > 25
        
        # Elder Ray signals
        bullish_elder = bull_power_smooth[i] > 0 and bull_power_smooth[i] > bear_power_smooth[i]
        bearish_elder = bear_power_smooth[i] < 0 and bear_power_smooth[i] < bull_power_smooth[i]
        
        price = close[i]
        
        if position == 0:
            # Long: bullish Alligator + strong trend + bullish Elder Ray
            if bullish_alligator and strong_trend and bullish_elder:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            # Short: bearish Alligator + strong trend + bearish Elder Ray
            elif bearish_alligator and strong_trend and bearish_elder:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 3 bars to reduce churn
            if bars_since_entry < 3:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Exit conditions
            if position == 1:
                # Exit long: bearish Alligator flip OR weak trend OR bearish Elder Ray
                if (not bullish_alligator) or (adx_aligned[i] < 20) or bearish_elder:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: bullish Alligator flip OR weak trend OR bullish Elder Ray
                if (not bearish_alligator) or (adx_aligned[i] < 20) or bullish_elder:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ADX_Alligator_ElderRay_Combo"
timeframe = "6h"
leverage = 1.0