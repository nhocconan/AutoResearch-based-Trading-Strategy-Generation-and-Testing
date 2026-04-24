#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator + Elder Ray with volume spike and chop regime filter.
- Williams Alligator (jaw=13, teeth=8, lips=5) defines trend direction via jaw-teeth-lips alignment.
- Elder Ray (bull/bear power) confirms trend strength using EMA13.
- Volume spike (>2.0x 20-bar average) filters for high-conviction moves.
- Choppiness Index (CHOP>61.8) avoids ranging markets; strategy only trades when CHOP<=61.8 (trending).
- Discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 20-50 trades/year (75-200 total over 4 years) to stay fee-efficient.
- Combines proven indicators from DB: Alligator, Elder Ray, volume, and chop regime.
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
    
    # Williams Alligator: SMA of median price (HL/2)
    median_price = (high + low) / 2.0
    jaw = pd.Series(median_price).rolling(window=13, min_periods=13).mean().values  # 13-bar
    teeth = pd.Series(median_price).rolling(window=8, min_periods=8).mean().values   # 8-bar
    lips = pd.Series(median_price).rolling(window=5, min_periods=5).mean().values    # 5-bar
    
    # Elder Ray: EMA13 for trend, Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume confirmation: > 2.0x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index: CHOP > 61.8 = ranging, CHOP <= 61.8 = trending
    # True Range = max(high-low, abs(high-prev_close), abs(low-prev_close))
    prev_close = np.append([close[0]], close[:-1])
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr1 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # CHOP = 100 * log10(sum(ATR14)/ (max(high)-min(low)) over 14 bars) / log10(14)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(pd.Series(atr1).rolling(window=14, min_periods=14).sum().values / (highest_high - lowest_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime filter: only trade when trending (CHOP <= 61.8)
        trending_regime = chop[i] <= 61.8
        # Volume confirmation
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Alligator bullish (lips > teeth > jaw) AND Bull Power > 0 AND volume + regime
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and bull_power[i] > 0 and volume_confirm and trending_regime:
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish (jaw > teeth > lips) AND Bear Power < 0 AND volume + regime
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and bear_power[i] < 0 and volume_confirm and trending_regime:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR Bull Power <= 0
            if jaw[i] > teeth[i] or teeth[i] > lips[i] or bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR Bear Power >= 0
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_Volume_Chop_v1"
timeframe = "4h"
leverage = 1.0