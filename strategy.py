#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + 1d volume spike + 1w trend filter.
# Long when price > Alligator Jaw (TEETH) AND volume > 2.0x 20-period 1d average AND 1w EMA50 > EMA100 (bullish weekly trend).
# Short when price < Alligator Jaw (TEETH) AND volume > 2.0x 20-period 1d average AND 1w EMA50 < EMA100 (bearish weekly trend).
# Exit when price crosses the Alligator Lips (LIPS).
# Uses discrete position size 0.25. Designed to catch trends with Alligator alignment and volume confirmation in both bull and bear markets.
# Target: 80-120 total trades over 4 years (20-30/year) to minimize fee drag while maintaining edge.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 6h Indicators: Williams Alligator (JAW=TEETH=13, TEETH=8, LIPS=5) ===
    # Alligator Jaw (blue line) - 13-period SMMA, shifted 8 bars ahead
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean()
    jaw = jaw.shift(8)  # shift 8 bars ahead
    # Alligator Teeth (red line) - 8-period SMMA, shifted 5 bars ahead
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean()
    teeth = teeth.shift(5)  # shift 5 bars ahead
    # Alligator Lips (green line) - 5-period SMMA, shifted 3 bars ahead
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean()
    lips = lips.shift(3)  # shift 3 bars ahead
    
    jaw_values = jaw.values
    teeth_values = teeth.values
    lips_values = lips.values
    
    # === 1d Indicators: Volume Spike (volume > 2.0x 20-period average) ===
    df_1d = get_htf_data(prices, '1d')
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    volume_spike = volume > (2.0 * vol_ma_1d_aligned)
    
    # === 1w Indicators: EMA50 > EMA100 (bullish trend) or EMA50 < EMA100 (bearish trend) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100_1w = pd.Series(close_1w).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    ema_100_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_100_1w)
    weekly_bullish = ema_50_1w_aligned > ema_100_1w_aligned
    weekly_bearish = ema_50_1w_aligned < ema_100_1w_aligned
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid (max 100 periods needed for EMA100)
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN or outside session
        if (np.isnan(jaw_values[i]) or np.isnan(teeth_values[i]) or np.isnan(lips_values[i]) or
            np.isnan(volume_spike[i]) or np.isnan(weekly_bullish[i]) or np.isnan(weekly_bearish[i]) or
            not session_filter[i]):
            signals[i] = 0.0
            position = 0
            continue
        
        # Current values
        price = close[i]
        vol_spike = volume_spike[i]
        is_weekly_bullish = weekly_bullish[i]
        is_weekly_bearish = weekly_bearish[i]
        
        # === EXIT LOGIC ===
        exit_signal = False
        
        if position == 1:  # Long position
            # Exit if price crosses below lips
            if price < lips_values[i]:
                exit_signal = True
        
        elif position == -1:  # Short position
            # Exit if price crosses above lips
            if price > lips_values[i]:
                exit_signal = True
        
        if exit_signal:
            signals[i] = 0.0
            position = 0
            continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price > Teeth AND volume spike AND weekly bullish trend
            if price > teeth_values[i] and vol_spike and is_weekly_bullish:
                signals[i] = 0.25
                position = 1
            
            # SHORT: Price < Teeth AND volume spike AND weekly bearish trend
            elif price < teeth_values[i] and vol_spike and is_weekly_bearish:
                signals[i] = -0.25
                position = -1
        
        else:
            signals[i] = position * 0.25
    
    return signals

name = "6h_Williams_Alligator_1dVolumeSpike_1wEMA_V1"
timeframe = "6h"
leverage = 1.0