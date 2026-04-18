#!/usr/bin/env python3
"""
4h Volume-Weighted Average Price (VWAP) Deviation with Volume Spike and ADX Trend Filter
Hypothesis: Price deviating significantly from VWAP (volume-weighted average price) on 4h,
combined with volume spikes (>2x average) and strong trend (ADX > 25), indicates mean-reversion
or momentum continuation depending on deviation direction. VWAP acts as dynamic support/resistance.
Designed to work in both bull and bear markets by capturing overextended moves.
Target: 20-30 trades/year to minimize fee drain.
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
    
    # Calculate VWAP for each bar (typical price * volume)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = typical_price * volume
    vwap_denominator = volume
    
    # Cumulative VWAP (reset daily? but we'll use rolling for simplicity and stability)
    # Using 20-period rolling VWAP to avoid look-ahead and stabilize
    vwap = pd.Series(vwap_numerator).rolling(window=20, min_periods=20).sum().values / \
           pd.Series(vwap_denominator).rolling(window=20, min_periods=20).sum().values
    
    # Calculate deviation from VWAP as percentage
    vwap_deviation = (close - vwap) / vwap * 100  # in percentage
    
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
    
    # Volume confirmation: volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ratio = volume / vol_ema
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Warmup for indicators (max of 20,20,14)
    
    for i in range(start_idx, n):
        if (np.isnan(vwap[i]) or np.isnan(vwap_deviation[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        dev = vwap_deviation[i]
        adx_val = adx[i]
        vol_conf = vol_ratio[i] > 2.0  # Volume spike filter
        
        if position == 0:
            # Strong trend and volume confirmation
            # Significant negative deviation (price below VWAP) = long (mean reversion)
            # Significant positive deviation (price above VWAP) = short (mean reversion)
            if adx_val > 25 and vol_conf:
                if dev < -1.5:  # Price more than 1.5% below VWAP
                    signals[i] = 0.25
                    position = 1
                elif dev > 1.5:  # Price more than 1.5% above VWAP
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit if deviation returns toward VWAP or trend weakens
            if dev > -0.5 or adx_val < 20:  # Price back within 0.5% of VWAP or weak trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit if deviation returns toward VWAP or trend weakens
            if dev < 0.5 or adx_val < 20:  # Price back within 0.5% of VWAP or weak trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_VWAP_Deviation_Volume_ADX"
timeframe = "4h"
leverage = 1.0