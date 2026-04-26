#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime
Hypothesis: 4h Camarilla R1/S1 breakout with volume spike, 1d EMA34 trend filter, and chop regime filter.
- Long when price breaks above Camarilla R1 (from prior 4h range) AND volume spike AND 1d EMA34 uptrend AND chop regime allows trending (CHOP < 61.8)
- Short when price breaks below Camarilla S1 AND volume spike AND 1d EMA34 downtrend AND chop regime allows trending (CHOP < 61.8)
- Uses prior 4h range for Camarilla levels (structure-based edge from prior completed 4h bar)
- Volume spike confirms institutional participation (2.0x 20-period average on 4h)
- 1d EMA34 filter ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
- Chop regime filter avoids whipsaws in sideways markets (only trade when CHOP < 61.8 = trending)
- Designed for moderate frequency (target 30-60 trades/year) to minimize fee drag
- Exit on opposite Camarilla level touch (S1 for longs, R1 for shorts) or trend reversal or chop regime shift
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for Camarilla levels (structure)
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate Camarilla levels from prior 4h bar (completed bar only)
    # Camarilla: R1 = close + (high-low)*1.1/12, S1 = close - (high-low)*1.1/12
    prior_4h_high = np.roll(df_4h['high'].values, 1)
    prior_4h_low = np.roll(df_4h['low'].values, 1)
    prior_4h_close = np.roll(df_4h['close'].values, 1)
    # First value is invalid due to roll
    prior_4h_high[0] = np.nan
    prior_4h_low[0] = np.nan
    prior_4h_close[0] = np.nan
    
    cam_r1 = prior_4h_close + (prior_4h_high - prior_4h_low) * 1.1 / 12
    cam_s1 = prior_4h_close - (prior_4h_high - prior_4h_low) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (no additional delay needed for structure)
    cam_r1_aligned = align_htf_to_ltf(prices, df_4h, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_4h, cam_s1)
    
    # Load 1d data ONCE before loop for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA34 for trend filter (needs completed 1d candle)
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    # Trend: 1 = uptrend (close > EMA34), -1 = downtrend (close < EMA34), 0 = neutral/invalid
    trend_1d = np.where(ema_34_1d_aligned > 0, 
                         np.where(close > ema_34_1d_aligned, 1, -1), 
                         0)
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    # Calculate Choppiness Index (CHOP) on 4h for regime filter
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP < 38.2 = strong trend, CHOP > 61.8 = sideways/chop, 38.2-61.8 = transitional
    # We only trade when CHOP < 61.8 (trending regime)
    tr14 = np.maximum(high - low, 
                      np.maximum(np.abs(high - np.roll(close, 1)), 
                                 np.abs(low - np.roll(close, 1))))
    tr14[0] = np.nan  # First TR is undefined
    atr14 = pd.Series(tr14).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr14).rolling(window=14, min_periods=14).sum().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range14 = highest_high14 - lowest_low14
    # Avoid division by zero
    chop = np.where(range14 > 0, 
                    100 * np.log10(sum_atr14) / np.log10(14) / np.log10(range14), 
                    100)  # Set to 100 (max choppy) when range is zero
    chop_regime = chop < 61.8  # True when market is trending (not choppy)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for prior 4h, 34 for 1d EMA, 14 for CHOP)
    start_idx = max(20, 1, 34, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(trend_1d[i]) or
            np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R1/S1 breakout conditions with volume confirmation, 1d trend filter, and chop regime
        if position == 0:
            # Long: Price breaks above Camarilla R1 AND volume spike AND 1d uptrend AND trending regime
            if close[i] > cam_r1_aligned[i] and volume_spike[i] and trend_1d[i] == 1 and chop_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S1 AND volume spike AND 1d downtrend AND trending regime
            elif close[i] < cam_s1_aligned[i] and volume_spike[i] and trend_1d[i] == -1 and chop_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S1 OR 1d trend turns down OR chop regime becomes choppy
            if close[i] < cam_s1_aligned[i] or trend_1d[i] == -1 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R1 OR 1d trend turns up OR chop regime becomes choppy
            if close[i] > cam_r1_aligned[i] or trend_1d[i] == 1 or not chop_regime[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0