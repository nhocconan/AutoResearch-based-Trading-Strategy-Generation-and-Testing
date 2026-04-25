#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2
Hypothesis: 12h Camarilla R1/S1 breakouts with 1d EMA50 trend filter, volume confirmation (>2.0x 20-period mean volume), and choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend). Designed for BTC/ETH with discrete sizing (0.25) to target 12-37 trades/year. Uses 1d HTF for trend and Camarilla levels, aligned via mtf_data. Avoids overtrading via tight entry conditions and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Camarilla levels from previous 1d bar (HLC of prior bar)
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * (1.0/4.0)  # R1 = C + 1.1*(H-L)/4
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * (1.0/4.0)  # S1 = C - 1.1*(H-L)/4
    
    # Align Camarilla levels to 12h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: >2.0x 20-period mean volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 2.0)
    
    # Choppiness regime filter on 12h timeframe
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest high - lowest low over period))
    # Simplified: CHOP > 61.8 = range (mean revert), CHOP < 38.2 = trending (trend follow)
    # We'll use trend following when CHOP < 38.2
    tr12 = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr12 = np.concatenate([[np.nan], tr12])
    atr14_12 = pd.Series(tr12).rolling(window=14, min_periods=14).mean().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop_raw = 100 * np.log10(atr14_12 * 14 / np.log10(highest_high - lowest_low + 1e-10))
    chop_raw = np.where((highest_high - lowest_low) > 0, chop_raw, 50.0)  # avoid division by zero
    chop_aligned = chop_raw  # already on 12h
    trend_regime = chop_aligned < 38.2  # trending regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA50, ATR, volume MA, CHOP
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Long: price breaks above Camarilla R1 in uptrend (price > 1d EMA50) with volume confirmation and trending regime
            # Short: price breaks below Camarilla S1 in downtrend (price < 1d EMA50) with volume confirmation and trending regime
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema50_1d_aligned[i]) and vol_confirm[i] and trend_regime[i]
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema50_1d_aligned[i]) and vol_confirm[i] and trend_regime[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below 1d EMA50 (trend reversal)
            exit_signal = close[i] < ema50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above 1d EMA50 (trend reversal)
            exit_signal = close[i] > ema50_1d_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike_Regime_v2"
timeframe = "12h"
leverage = 1.0