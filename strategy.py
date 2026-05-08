#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d EMA Trend Filter + Volume Spike
# Uses 12h Williams Alligator (Jaw/Teeth/Lips) to identify trend direction,
# 1d EMA50 for higher timeframe trend filter, and volume spikes (>2x average)
# for entry timing. Designed to capture strong trends while avoiding whipsaws
# in both bull and bear markets. Target: 15-30 trades/year.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
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
    
    # Get daily data for EMA trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full(len(close_daily), np.nan)
    if len(close_daily) >= 50:
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = (close_daily[i] * 2 + ema50_daily[i-1] * 48) / 50
    
    # Calculate 12h Williams Alligator (13,8,5 SMAs with 8,5,3 offsets)
    # Jaw: 13-period SMA, shifted 8 bars
    # Teeth: 8-period SMA, shifted 5 bars  
    # Lips: 5-period SMA, shifted 3 bars
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    jaw = np.full_like(close, np.nan)
    teeth = np.full_like(close, np.nan)
    lips = np.full_like(close, np.nan)
    
    # Apply shifts: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    for i in range(8, len(jaw)):
        jaw[i] = jaw_raw[i-8]
    for i in range(5, len(teeth)):
        teeth[i] = teeth_raw[i-5]
    for i in range(3, len(lips)):
        lips[i] = lips_raw[i-3]
    
    # Calculate 12h volume average for volume spike detection
    vol_avg_20 = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(20, len(volume)):
            vol_avg_20[i] = np.mean(volume[i-20:i])
    
    # Align daily EMA50 to 12h timeframe
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 13)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(ema50_daily_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2x 20-period average
        vol_spike = volume[i] > 2.0 * vol_avg_20[i]
        
        if position == 0:
            # Look for entry: Alligator aligned (Lips > Teeth > Jaw for long, Lips < Teeth < Jaw for short)
            # plus EMA trend filter and volume spike
            alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long when bullish alignment, price above daily EMA50, and volume spike
            long_condition = (
                alligator_long and
                close[i] > ema50_daily_aligned[i] and
                vol_spike
            )
            
            # Short when bearish alignment, price below daily EMA50, and volume spike
            short_condition = (
                alligator_short and
                close[i] < ema50_daily_aligned[i] and
                vol_spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator turns bearish or price closes below Jaw
            if lips[i] < teeth[i] or close[i] < jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator turns bullish or price closes above Jaw
            if lips[i] > teeth[i] or close[i] > jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals