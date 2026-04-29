#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume spike
# Williams Alligator (jaw/teeth/lips) identifies trend absence when lines intertwine
# Breakout above lips (bullish) or below jaw (bearish) with alignment indicates strong momentum
# 1d Elder Ray (Bear/Bull Power) confirms intermediate-term trend direction
# Volume spike confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag on 12h timeframe
# Works in both bull and bear markets by following 1d trend while capturing 12h momentum breakouts

name = "12h_WilliamsAlligator_1dElderRay_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1d calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray: Bear Power = Low - EMA13, Bull Power = High - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bear_power_1d = low_1d - ema13_1d   # Negative = bearish bias
    bull_power_1d = high_1d - ema13_1d  # Positive = bullish bias
    
    # Align 1d Elder Ray to 12h timeframe (completed 1d bar only)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    # Jaw (blue): 13-period SMMA smoothed by 8 periods
    # Teeth (red): 8-period SMMA smoothed by 5 periods  
    # Lips (green): 5-period SMMA smoothed by 3 periods
    # SMMA = Smoothed Moving Average (similar to Wilder's MA)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA(i) = (SMMA(i-1)*(period-1) + arr[i]) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    jaw = smma(jaw, 8)     # smoothed by 8
    
    teeth = smma(close, 8)   # 8-period SMMA
    teeth = smma(teeth, 5)   # smoothed by 5
    
    lips = smma(close, 5)    # 5-period SMMA
    lips = smma(lips, 3)     # smoothed by 3
    
    # Volume confirmation: volume > 2.0x 20-period average (~10 days)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 20)  # warmup for Alligator and volume MA
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if (np.isnan(bear_power_aligned[i]) or np.isnan(bull_power_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i])):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_jaw = jaw[i]
        curr_teeth = teeth[i]
        curr_lips = lips[i]
        curr_bear_power = bear_power_aligned[i]
        curr_bull_power = bull_power_aligned[i]
        curr_volume_confirm = volume_confirm[i]
        
        # Alligator trend conditions:
        # Bullish alignment: Lips > Teeth > Jaw (green above red above blue)
        # Bearish alignment: Jaw > Teeth > Lips (blue above red above green)
        is_bullish_alligator = curr_lips > curr_teeth > curr_jaw
        is_bearish_alligator = curr_jaw > curr_teeth > curr_lips
        
        # Elder Ray trend filter:
        # Bullish regime: Bull Power > 0 AND Bear Power < 0 (clear bullish bias)
        # Bearish regime: Bear Power < 0 AND Bull Power < 0 (clear bearish bias)
        is_bullish_regime = curr_bull_power > 0 and curr_bear_power < 0
        is_bearish_regime = curr_bear_power < 0 and curr_bull_power < 0
        
        if position == 0:  # Flat - look for new entries
            # Only trade with volume confirmation
            if curr_volume_confirm:
                # Bullish entry: price breaks above lips AND bullish alignment AND bullish regime
                if curr_high > curr_lips and is_bullish_alligator and is_bullish_regime:
                    signals[i] = 0.25
                    position = 1
                # Bearish entry: price breaks below jaw AND bearish alignment AND bearish regime
                elif curr_low < curr_jaw and is_bearish_alligator and is_bearish_regime:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:  # Long position - exit conditions
            # Exit when: price breaks below teeth (weaker bullish signal) OR Alligator turns bearish
            if curr_low < curr_teeth or not is_bullish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit conditions
            # Exit when: price breaks above teeth (weaker bearish signal) OR Alligator turns bullish
            if curr_high > curr_teeth or not is_bearish_alligator:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals