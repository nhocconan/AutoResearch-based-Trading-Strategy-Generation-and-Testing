#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d ATR Regime Filter and Volume Spike.
- Primary timeframe: 12h for execution, HTF: 1w for regime, 1d for ATR and Williams Alligator.
- Entry: Williams Alligator signals bullish (jaw < teeth < lips) or bearish (jaw > teeth > lips) on 12h close, with volume > 2.0x 20-period volume MA.
- Regime filter: only trade when 1d ATR(14) > 1.5 * 50-period ATR MA (high volatility regime).
- Williams Alligator uses smoothed medians (SMMA) of 13, 8, 5 periods with 8, 5, 3 shifts.
- Volume confirmation reduces false signals in low volatility.
- Exit: Opposite Alligator signal or volatility contraction (ATR ratio < 1.0).
- Discrete signal size: 0.25 to balance return and drawdown control.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
- Works in bull via buying Alligator alignment in uptrend, in bear via selling alignment in downtrend.
- Williams Alligator catches trends early; volatility filter avoids choppy markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(source, length):
    """Smoothed Moving Average (SMMA) - also called Wilder's MA or RMA"""
    if length <= 0:
        return source.copy()
    result = np.full_like(source, np.nan, dtype=np.float64)
    if len(source) < length:
        return result
    # First value is simple average
    result[length-1] = np.mean(source[:length])
    # Subsequent values: SMMA = (PREV_SMMA * (length-1) + CURRENT) / length
    for i in range(length, len(source)):
        result[i] = (result[i-1] * (length-1) + source[i]) / length
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for volatility regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First TR is undefined
    
    # ATR(14) using Wilder's smoothing (SMMA)
    atr_14_1d = smma(tr, 14)
    # 50-period ATR MA for regime comparison
    atr_ma_50_1d = pd.Series(atr_14_1d).rolling(window=50, min_periods=50).mean().values
    # Regime: high volatility when ATR > 1.5 * ATR_MA
    volatility_regime = atr_14_1d > (1.5 * atr_ma_50_1d)
    volatility_regime_aligned = align_htf_to_ltf(prices, df_1d, volatility_regime.astype(float))
    
    # Calculate 1d Williams Alligator (jaw=13, teeth=8, lips=5)
    median_1d = (high_1d + low_1d) / 2
    jaw = smma(median_1d, 13)  # Blue line
    teeth = smma(median_1d, 8)   # Red line
    lips = smma(median_1d, 5)    # Green line
    
    # Shift: jaw 8 bars, teeth 5 bars, lips 3 bars (Alligator sleeps with mouth closed)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # First values after shift are invalid
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (completed 1d bar only)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume > 2.0 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13, 8, 5) + 8  # Need ATR MA(50), volume MA(20), Alligator shifts
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(volatility_regime_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for Alligator signals (only in high volatility regime)
        if volatility_regime_aligned[i] > 0.5 and volume_spike[i]:
            bullish_aligned = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
            bearish_aligned = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
            
            if position == 0:
                # Enter long on bullish Alligator alignment
                if bullish_aligned:
                    signals[i] = 0.25
                    position = 1
                # Enter short on bearish Alligator alignment
                elif bearish_aligned:
                    signals[i] = -0.25
                    position = -1
            elif position == 1:
                # Exit long on bearish Alligator alignment or volatility contraction
                if bearish_aligned or volatility_regime_aligned[i] <= 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short on bullish Alligator alignment or volatility contraction
                if bullish_aligned or volatility_regime_aligned[i] <= 0.5:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
        else:
            # Low volatility or no volume spike: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Williams_Alligator_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0