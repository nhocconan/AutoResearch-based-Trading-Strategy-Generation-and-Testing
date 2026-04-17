#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1w Trend Filter + Volume Spike
Long when 6h Bull Power > 0 (close > EMA13) AND 1w EMA34 uptrend (close > EMA34) AND volume spike > 2x 20-period average.
Short when 6h Bear Power < 0 (close < EMA13) AND 1w EMA34 downtrend (close < EMA34) AND volume spike > 2x 20-period average.
Exit when Elder Power reverses OR volume drops below average.
Uses 1w for major trend filter, 6h for Elder Ray signals and volume confirmation.
Designed to capture strong momentum moves in alignment with weekly trend, working in both bull and bear markets by filtering counter-trend signals.
Target: 12-25 trades/year per symbol to minimize fee drag on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_to_ltf, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Calculate 6h EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13_6h = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = close - ema13_6h  # Bull Power = Close - EMA13
    bear_power = close - ema13_6h  # Bear Power = Close - EMA13 (same calculation, different interpretation)
    
    # Calculate 6h volume MA20 for volume spike detection
    volume_s = pd.Series(volume)
    vol_ma_20_6h = volume_s.rolling(window=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for indicators to warm up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(ema13_6h[i]) or
            np.isnan(vol_ma_20_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 2x 20-period average
        volume_spike = not np.isnan(volume[i]) and \
                      not np.isnan(vol_ma_20_6h[i]) and \
                      volume[i] > 2.0 * vol_ma_20_6h[i]
        
        # 1w trend filter
        uptrend_1w = close[i] > ema34_1w_aligned[i]
        downtrend_1w = close[i] < ema34_1w_aligned[i]
        
        # Elder Ray signals
        bull_signal = bull_power[i] > 0  # Close > EMA13
        bear_signal = bear_power[i] < 0  # Close < EMA13
        
        if position == 0:
            # Long: Bull Power positive AND 1w uptrend AND volume spike
            if (bull_signal and uptrend_1w and volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative AND 1w downtrend AND volume spike
            elif (bear_signal and downtrend_1w and volume_spike):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bull Power turns negative OR volume drops below average
            volume_normal = not np.isnan(volume[i]) and \
                           not np.isnan(vol_ma_20_6h[i]) and \
                           volume[i] < vol_ma_20_6h[i]
            if (not bull_signal or volume_normal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bear Power turns positive OR volume drops below average
            volume_normal = not np.isnan(volume[i]) and \
                           not np.isnan(vol_ma_20_6h[i]) and \
                           volume[i] < vol_ma_20_6h[i]
            if (not bear_signal or volume_normal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1wEMA34Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0