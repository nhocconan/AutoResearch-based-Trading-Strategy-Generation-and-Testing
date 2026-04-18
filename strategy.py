#!/usr/bin/env python3
"""
4h Three-Bar Reversal with Volume Surge and ADX Filter
Hypothesis: Three-bar reversal patterns (three consecutive higher highs/lows) signal momentum exhaustion.
Combined with volume surge (>2x average) and weak trend (ADX < 25) to catch reversals in both bull and bear markets.
Target: 20-40 trades/year to minimize fee drain.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # ADX for trend strength (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    di_plus = np.where(tr14 > 0, 100 * dm_plus14 / tr14, 0)
    di_minus = np.where(tr14 > 0, 100 * dm_minus14 / tr14, 0)
    
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume surge: volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > (2 * vol_ema)
    
    # Three-bar reversal detection
    # Bullish reversal: three consecutive higher lows
    bull_reversal = (low > np.roll(low, 1)) & (np.roll(low, 1) > np.roll(low, 2)) & (np.roll(low, 2) > np.roll(low, 3))
    # Bearish reversal: three consecutive lower highs
    bear_reversal = (high < np.roll(high, 1)) & (np.roll(high, 1) < np.roll(high, 2)) & (np.roll(high, 2) < np.roll(high, 3))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 14,20,3)
    
    for i in range(start_idx, n):
        if (np.isnan(adx[i]) or np.isnan(vol_surge[i]) or 
            np.isnan(bull_reversal[i]) or np.isnan(bear_reversal[i])):
            signals[i] = 0.0
            continue
        
        adx_val = adx[i]
        vol_surge_val = vol_surge[i]
        bull_rev = bull_reversal[i]
        bear_rev = bear_reversal[i]
        
        if position == 0:
            # Look for bullish reversal with volume surge and weak trend
            if bull_rev and vol_surge_val and adx_val < 25:
                signals[i] = 0.25
                position = 1
            # Look for bearish reversal with volume surge and weak trend
            elif bear_rev and vol_surge_val and adx_val < 25:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit on bearish reversal or trend strengthening
            if bear_rev or adx_val > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit on bullish reversal or trend strengthening
            if bull_rev or adx_val > 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_ThreeBar_Reversal_Volume_ADX"
timeframe = "4h"
leverage = 1.0