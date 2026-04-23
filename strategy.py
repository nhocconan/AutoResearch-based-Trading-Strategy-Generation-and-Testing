#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation
- Bull Power = High - EMA13, Bear Power = EMA13 - Low
- Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND price > 1d EMA50 (uptrend) AND volume > 1.5x 20-period average
- Short when Bear Power > 0 AND Bull Power < 0 (bearish momentum) AND price < 1d EMA50 (downtrend) AND volume > 1.5x 20-period average
- Exit when momentum diverges: Bull Power < 0 for long OR Bear Power < 0 for short
- Uses 1d EMA50 for HTF trend alignment to avoid counter-trend entries
- Volume confirmation ensures institutional participation and reduces false signals
- Elder Ray measures bull/bear power relative to EMA13, effective in both trending and ranging markets
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag
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
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA13 for Elder Ray (primary timeframe)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14)  # Need 20 for volume MA, 50 for EMA50, 14 for EMA13
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray conditions
        bullish_momentum = bull_power[i] > 0 and bear_power[i] < 0  # Bull Power > 0 AND Bear Power < 0
        bearish_momentum = bear_power[i] > 0 and bull_power[i] < 0  # Bear Power > 0 AND Bull Power < 0
        
        # Trend filter (using 1d EMA50)
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: bullish momentum + uptrend + volume confirmation
            if bullish_momentum and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: bearish momentum + downtrend + volume confirmation
            elif bearish_momentum and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: momentum divergence
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power < 0 (loss of bullish momentum)
                if bull_power[i] < 0:
                    exit_signal = True
            elif position == -1:
                # Exit short: Bear Power < 0 (loss of bearish momentum)
                if bear_power[i] < 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_ElderRay_1dEMA50_Trend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0