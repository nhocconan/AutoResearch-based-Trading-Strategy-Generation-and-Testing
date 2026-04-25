#!/usr/bin/env python3
"""
12h Williams Alligator + Daily EMA34 Trend + Volume Spike
Hypothesis: Williams Alligator (jaw/teeth/lips) identifies trend absence (all lines intertwined) vs presence (lines diverged). 
Breakouts above/below the Alligator with daily EMA34 trend alignment and volume spike capture strong moves.
Works in bull markets (trend continuation) and bear markets (sharp reversals from consolidation).
Uses daily HTF data loaded ONCE before loop.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # ATR for stop (optional, using signal=0 for exit)
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Daily EMA34 trend filter (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe (using prices directly as we are on 12h timeframe)
    # Alligator: Jaw (13-period SMMA, offset 8), Teeth (8-period SMMA, offset 5), Lips (5-period SMMA, offset 3)
    def smma(source, period):
        """Smoothed Moving Average"""
        result = np.full_like(source, np.nan, dtype=float)
        if len(source) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(source[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CURRENT_VALUE) / PERIOD
        for i in range(period, len(source)):
            result[i] = (result[i-1] * (period-1) + source[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Alligator is "sleeping" (no trend) when lines are intertwined
    # We consider it "awake" (trending) when lips > teeth > jaw (uptrend) or lips < teeth < jaw (downtrend)
    # For simplicity, we'll use the condition that lips is outside the jaw-teeth range
    alligator_awake = (lips > jaw) & (lips < teeth) | (lips < jaw) & (lips > teeth)
    alligator_awake = ~alligator_awake  # True when awake (lines diverged)
    
    # Trend direction based on Alligator alignment
    # Uptrend: lips > teeth > jaw
    # Downtrend: lips < teeth < jaw
    alligator_uptrend = (lips > teeth) & (teeth > jaw)
    alligator_downtrend = (lips < teeth) & (teeth < jaw)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for Alligator (max period 13 + offsets) and EMA/volume
    start_idx = max(35, 20, 13) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        vol_spike = volume_spike[i]
        
        # Alligator signals
        awake = alligator_awake[i]
        uptrend = alligator_uptrend[i]
        downtrend = alligator_downtrend[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator awake + volume spike + daily EMA34 trend alignment
            long_entry = awake and uptrend and vol_spike and (curr_close > ema_34_1d_aligned[i])
            short_entry = awake and downtrend and vol_spike and (curr_close < ema_34_1d_aligned[i])
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on Alligator sleep (trend weakening) or trend change
            if not awake or curr_close < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit on Alligator sleep (trend weakening) or trend change
            if not awake or curr_close > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0