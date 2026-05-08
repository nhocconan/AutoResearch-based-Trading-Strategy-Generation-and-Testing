#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1w EMA50 Trend Filter + Volume Spike
# Uses weekly EMA50 for trend direction, Williams Alligator (Jaws/Teeth/Lips) on 12h for entry signals,
# and volume spike (>1.5x average) for confirmation. Designed to capture trends in both bull and bear markets
# by following the weekly trend while using the Alligator's convergence/divergence to avoid whipsaws.
# Target: 15-35 trades/year on 12h timeframe.

name = "12h_WilliamsAlligator_1wEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_weekly = df_weekly['close'].values
    ema50_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 50:
        ema50_weekly[49] = np.mean(close_weekly[:50])
        for i in range(50, len(close_weekly)):
            ema50_weekly[i] = (close_weekly[i] * 2 + ema50_weekly[i-1] * 48) / 50
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator components (SMMA with periods 13,8,5 and shifts 8,5,3)
    # Jaw (13-period SMMA, 8 periods ahead)
    close_12h = df_12h['close'].values
    smma_13 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 13:
        smma_13[12] = np.mean(close_12h[:13])
        for i in range(13, len(close_12h)):
            smma_13[i] = (smma_13[i-1] * 12 + close_12h[i]) / 13
    jaws = np.full(len(close_12h), np.nan)
    if len(smma_13) >= 21:  # 13 + 8
        jaws[20:] = smma_13[12:-8]  # shift 8 bars ahead
    
    # Teeth (8-period SMMA, 5 periods ahead)
    smma_8 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 8:
        smma_8[7] = np.mean(close_12h[:8])
        for i in range(8, len(close_12h)):
            smma_8[i] = (smma_8[i-1] * 7 + close_12h[i]) / 8
    teeth = np.full(len(close_12h), np.nan)
    if len(smma_8) >= 13:  # 8 + 5
        teeth[12:] = smma_8[7:-5]  # shift 5 bars ahead
    
    # Lips (5-period SMMA, 3 periods ahead)
    smma_5 = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 5:
        smma_5[4] = np.mean(close_12h[:5])
        for i in range(5, len(close_12h)):
            smma_5[i] = (smma_5[i-1] * 4 + close_12h[i]) / 5
    lips = np.full(len(close_12h), np.nan)
    if len(smma_5) >= 8:  # 5 + 3
        lips[7:] = smma_5[4:-3]  # shift 3 bars ahead
    
    # Calculate 12h volume average for volume spike detection
    vol_12h = df_12h['volume'].values
    vol_avg_20_12h = np.full(len(vol_12h), np.nan)
    if len(vol_12h) >= 20:
        for i in range(20, len(vol_12h)):
            vol_avg_20_12h[i] = np.mean(vol_12h[i-20:i])
    
    # Align weekly and 12h indicators to 12h timeframe (which is our base)
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    vol_avg_20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_avg_20_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 13, 8, 5, 20)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(jaws_aligned[i]) or
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(vol_avg_20_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current 12h volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_avg_20_12h_aligned[i]
        
        if position == 0:
            # Look for entry: Alligator alignment with weekly trend and volume spike
            # Alligator is aligned when Lips > Teeth > Jaws (bullish) or Lips < Teeth < Jaws (bearish)
            bullish_aligned = (lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i])
            bearish_aligned = (lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i])
            
            # Long when bullish alignment, above weekly EMA50, and volume spike
            long_condition = (
                bullish_aligned and
                close[i] > ema50_weekly_aligned[i] and
                vol_spike
            )
            
            # Short when bearish alignment, below weekly EMA50, and volume spike
            short_condition = (
                bearish_aligned and
                close[i] < ema50_weekly_aligned[i] and
                vol_spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator loses alignment or price crosses below weekly EMA50
            if not (lips_aligned[i] > teeth_aligned[i] > jaws_aligned[i]) or close[i] < ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses alignment or price crosses above weekly EMA50
            if not (lips_aligned[i] < teeth_aligned[i] < jaws_aligned[i]) or close[i] > ema50_weekly_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals