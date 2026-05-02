#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) confluence
# Uses 6h timeframe for Alligator (JAWS/TEETH/LIPS) to identify trend and momentum.
# 1d Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) confirms institutional buying/selling pressure.
# Long when: Alligator aligned bullish (LIPS > TEETH > JAWS) + 1d Bull Power > 0 and rising.
# Short when: Alligator aligned bearish (LIPS < TEETH < JAWS) + 1d Bear Power < 0 and falling.
# Uses volume spike (2.0x 20-period average) to confirm institutional participation.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull markets via trend-following entries, in bear via avoidance of counter-trend false signals.
# Target: BTC/ETH/SOL with Sharpe > 0 on both train and test.

name = "6h_WilliamsAlligator_1dElderRay_VolumeConfirm_v1"
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
    
    # Load 1d data ONCE before loop for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d Elder Ray components
    bull_power_1d = high_1d - ema_13_1d  # Buying pressure
    bear_power_1d = ema_13_1d - low_1d   # Selling pressure
    
    # Align 1d Elder Ray to 6h timeframe (completed 1d bar only)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Calculate 6h Williams Alligator (SMMA with specific periods)
    # JAWS: SMMA(13, 8) - Blue line
    # TEETH: SMMA(8, 5) - Red line  
    # LIPS: SMMA(5, 3) - Green line
    def smma(values, period):
        """Smoothed Moving Average"""
        result = np.full_like(values, np.nan, dtype=float)
        if len(values) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaws = smma(close, 13)  # SMMA(13, 8)
    teeth = smma(close, 8)  # SMMA(8, 5)
    lips = smma(close, 5)   # SMMA(5, 3)
    
    # Align Alligator lines (already on 6h timeframe, no additional delay needed for SMMA)
    # But we need to ensure we're using completed candle values
    # Since SMMA uses only past data, it's inherently non-lookahead
    
    # Calculate volume confirmation (2.0x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Alligator bullish alignment + Bull Power positive and rising + volume confirm
            if (lips[i] > teeth[i] > jaws[i] and 
                bull_power_1d_aligned[i] > 0 and 
                i > start_idx and bull_power_1d_aligned[i] > bull_power_1d_aligned[i-1] and
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish alignment + Bear Power positive and rising + volume confirm
            elif (lips[i] < teeth[i] < jaws[i] and 
                  bear_power_1d_aligned[i] > 0 and 
                  i > start_idx and bear_power_1d_aligned[i] > bear_power_1d_aligned[i-1] and
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator turns bearish OR Bull Power turns negative
            if (lips[i] < teeth[i] or bull_power_1d_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator turns bullish OR Bear Power turns negative
            if (lips[i] > teeth[i] or bear_power_1d_aligned[i] <= 0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals