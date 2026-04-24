#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d ATR regime filter + volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ATR-based regime filter.
- Donchian breakout: Long when price > highest high of last 20 periods, Short when price < lowest low.
- Regime filter: ATR(14) ratio > 1.2 = high volatility (trade breakouts), ATR ratio < 0.8 = low volatility (fade breakouts).
- Volume confirmation: current volume > 1.3x 20-period volume MA to ensure participation.
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
- Works in bull via buying breakouts in high volatility, in bear via selling breakdowns in high volatility, and fading breakouts in low volatility/chop.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d ATR(14) for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR(14)
    atr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # ATR(50) for ratio
    atr_50 = pd.Series(tr).ewm(alpha=1/50, adjust=False, min_periods=50).mean().values
    
    # ATR ratio: short-term / long-term volatility
    atr_ratio = np.where(atr_50 > 0, atr_14 / atr_50, 1.0)
    
    # Align 1d ATR ratio to 4h timeframe (completed 1d bar only)
    atr_ratio_aligned = align_htf_to_ltf(prices, df_1d, atr_ratio)
    
    # Donchian channels on 4h
    donchian_window = 20
    highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(donchian_window, 50, 20)  # Donchian(20) + ATR(50) + volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(atr_ratio_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Regime: ATR ratio > 1.2 = high volatility (trade breakouts), < 0.8 = low volatility (fade breakouts)
            if atr_ratio_aligned[i] > 1.2:
                # High volatility regime: trade breakouts in price direction
                if close[i] > highest_high[i] and volume_spike[i]:
                    # Upside breakout with volume: go long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < lowest_low[i] and volume_spike[i]:
                    # Downside breakdown with volume: go short
                    signals[i] = -0.25
                    position = -1
            elif atr_ratio_aligned[i] < 0.8:
                # Low volatility regime: fade breakouts (mean reversion)
                if close[i] > highest_high[i] and volume_spike[i]:
                    # Price broke above Donchian high in low vol: sell expecting reversion
                    signals[i] = -0.25
                    position = -1
                elif close[i] < lowest_low[i] and volume_spike[i]:
                    # Price broke below Donchian low in low vol: buy expecting reversion
                    signals[i] = 0.25
                    position = 1
        elif position == 1:
            # Long exit: price returns to midline or opposite breakout
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] < midline or close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to midline or opposite breakout
            midline = (highest_high[i] + lowest_low[i]) / 2
            if close[i] > midline or close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dATR_Ratio_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0