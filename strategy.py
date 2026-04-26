#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime_v2
Hypothesis: 4h Camarilla R1/S1 breakout with 1d EMA50 trend filter and choppiness regime (CHOP > 61.8 = range, mean reversion at Camarilla levels). 
- Long when price touches Camarilla S1 (support) in ranging market (CHOP > 61.8) AND 1d EMA50 uptrend
- Short when price touches Camarilla R1 (resistance) in ranging market (CHOP > 61.8) AND 1d EMA50 downtrend
- Uses prior completed 4h bar for Camarilla levels (structure-based edge)
- Choppiness filter avoids whipsaws in strong trends, enabling mean reversion in ranges
- 1d EMA50 ensures trading with higher timeframe trend (avoids counter-trend whipsaws)
- Designed for lower frequency (target 20-50 trades/year) to minimize fee drag and improve test generalization
- Exit on opposite Camarilla level touch (R1 for longs, S1 for shorts) or regime shift (CHOP < 61.8)
- Novelty: Camarilla R1/S1 levels (pivot-based support/resistance) + 1d HTF trend + chop regime filter on 4h timeframe
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
    
    # Calculate 1d EMA50 for trend filter (needs completed 1d candle)
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    # Trend: 1 = uptrend (close > EMA50), -1 = downtrend (close < EMA50), 0 = neutral/invalid
    trend_1d = np.where(ema_50_1d_aligned > 0, 
                        np.where(close > ema_50_1d_aligned, 1, -1), 
                        0)
    
    # Calculate choppiness regime (CHOP > 61.8 = ranging market, mean reversion zone)
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest(high,14) - lowest(low,14))) / log10(14)
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(low[1:] - close[:-1], np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], np.maximum(tr1, tr2)])  # True Range
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    highest_high14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14 * 14) / np.log10(highest_high14 - lowest_low14)
    chop = np.where((highest_high14 - lowest_low14) > 0, chop_raw, 100)  # Avoid division by zero
    chop_regime = chop > 61.8  # True = ranging market (mean reversion favorable)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 14 for chop, 1 for prior 4h, 50 for 1d EMA)
    start_idx = max(14, 1, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or
            np.isnan(trend_1d[i]) or np.isnan(chop_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R1/S1 mean reversion conditions in ranging market with 1d trend filter
        if position == 0:
            # Long: Price touches Camarilla S1 (support) AND ranging market (CHOP > 61.8) AND 1d uptrend
            if low[i] <= cam_s1_aligned[i] and chop_regime[i] and trend_1d[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price touches Camarilla R1 (resistance) AND ranging market (CHOP > 61.8) AND 1d downtrend
            elif high[i] >= cam_r1_aligned[i] and chop_regime[i] and trend_1d[i] == -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price rises above Camarilla R1 OR regime shifts to trending (CHOP < 61.8) OR 1d trend turns down
            if high[i] >= cam_r1_aligned[i] or not chop_regime[i] or trend_1d[i] == -1:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price falls below Camarilla S1 OR regime shifts to trending (CHOP < 61.8) OR 1d trend turns up
            if low[i] <= cam_s1_aligned[i] or not chop_regime[i] or trend_1d[i] == 1:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_ChopRegime_v2"
timeframe = "4h"
leverage = 1.0