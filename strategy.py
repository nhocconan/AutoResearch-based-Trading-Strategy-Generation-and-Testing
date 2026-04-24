#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d ATR regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1d for ATR regime and Donchian levels.
- Donchian channels calculated from previous 20 periods on 1d timeframe.
- ATR regime: CHOP > 61.8 = ranging (mean revert at Donchian bounds), CHOP < 38.2 = trending (breakout in direction of trend).
- Volume confirmation: current 12h volume > 1.5 * 20-period volume MA.
- Entry: Long when price breaks above Donchian upper in trending regime OR mean reverts from lower in ranging regime.
         Short when price breaks below Donchian lower in trending regime OR mean reverts from upper in ranging regime.
- Exit: Opposite Donchian level touch or regime change.
- Works in bull via buying breakouts/mean reversions, in bear via selling breakdowns/mean reversions.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf, compute_williams_fractals

def calculate_chop(high, low, close, window=14):
    """Calculate Choppiness Index"""
    if len(high) < window:
        return np.full(len(high), np.nan)
    atr_sum = 0
    for i in range(len(high)):
        if i == 0:
            tr = high[i] - low[i]
        else:
            tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_sum += tr
    atr = atr_sum / window
    max_high = pd.Series(high).rolling(window=window, min_periods=window).max().values
    min_low = pd.Series(low).rolling(window=window, min_periods=window).min().values
    chop = 100 * np.log10(atr * window / (max_high - min_low)) / np.log10(window)
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR regime and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR for Chop (using TR)
    tr_1d = np.zeros(len(df_1d))
    for i in range(len(df_1d)):
        if i == 0:
            tr_1d[i] = df_1d['high'].iloc[i] - df_1d['low'].iloc[i]
        else:
            tr_1d[i] = max(
                df_1d['high'].iloc[i] - df_1d['low'].iloc[i],
                abs(df_1d['high'].iloc[i] - df_1d['close'].iloc[i-1]),
                abs(df_1d['low'].iloc[i] - df_1d['close'].iloc[i-1])
            )
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d Chop
    chop_1d = calculate_chop(df_1d['high'].values, df_1d['low'].values, df_1d['close'].values, window=14)
    
    # Align 1d indicators to 12h
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Volume confirmation: current 12h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # Need enough 1d bars for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime-based entry logic
            if chop_aligned[i] < 38.2:  # Trending regime
                # Breakout entry
                if volume_spike[i]:
                    if close[i] > donchian_high_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    elif close[i] < donchian_low_aligned[i]:
                        signals[i] = -0.25
                        position = -1
            else:  # Ranging regime (CHOP > 61.8) or transition
                # Mean reversion at Donchian bounds
                if volume_spike[i]:
                    if close[i] <= donchian_low_aligned[i] * 1.001:  # Near lower bound
                        signals[i] = 0.25
                        position = 1
                    elif close[i] >= donchian_high_aligned[i] * 0.999:  # Near upper bound
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: touch opposite Donchian level or regime shift to ranging
            if close[i] >= donchian_low_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: touch opposite Donchian level or regime shift to ranging
            if close[i] <= donchian_high_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dATRRegime_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0