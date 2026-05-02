#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA34 trend filter and volume confirmation
# Williams Alligator (Jaw=TEETH=LIPS) identifies trend absence (alligator sleeping) vs presence (awake).
# Long when LIPS > TEETH > JAW (bullish alignment) + price above 1d EMA34 + volume spike.
# Short when LIPS < TEETH < JAW (bearish alignment) + price below 1d EMA34 + volume spike.
# Uses 12h timeframe to reduce trade frequency (target: 12-37 trades/year) and avoid fee drag.
# Alligator smoothed with SMMA (similar to Wilder's smoothing) for reliable trend signals.
# Works in both bull and bear markets by trading with 1d trend direction during awakened phases.

name = "12h_WilliamsAlligator_1dEMA34_Trend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1d for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw = Smoothed Median Price (13-period, 8-bar shift)
    # Teeth = Smoothed Median Price (8-period, 5-bar shift)
    # Lips = Smoothed Median Price (5-period, 3-bar shift)
    median_price = (high + low) / 2
    
    def smma(arr, period):
        """Smoothed Moving Average (similar to Wilder's smoothing)"""
        result = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)  # Jaw (Blue)
    teeth = smma(median_price, 8)  # Teeth (Red)
    lips = smma(median_price, 5)   # Lips (Green)
    
    # Volume confirmation (2.0x 24-period average) on 12h
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for Alligator and EMA calculations)
    start_idx = 50  # max(13 for Jaw, 34 for EMA, 24 for volume +1 for shift)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Bullish Alligator alignment: Lips > Teeth > Jaw (alligator awake, biting up)
            bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaw[i]
            # Bearish Alligator alignment: Lips < Teeth < Jaw (alligator awake, biting down)
            bearish_alignment = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long entry: Bullish alignment + price above 1d EMA34 + volume spike
            if bullish_alignment and close[i] > ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Bearish alignment + price below 1d EMA34 + volume spike
            elif bearish_alignment and close[i] < ema_34_1d_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Alligator alignment breaks (Lips < Teeth or Teeth < Jaw) or price crosses below 1d EMA34
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Alligator alignment breaks (Lips > Teeth or Teeth > Jaw) or price crosses above 1d EMA34
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals