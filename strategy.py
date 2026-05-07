#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation.
# Long when: Jaw < Teeth < Lips (bullish alignment) AND 12h EMA50 rising AND volume > 1.5 * EMA20(volume).
# Short when: Jaw > Teeth > Lips (bearish alignment) AND 12h EMA50 falling AND volume > 1.5 * EMA20(volume).
# Exit when Alligator alignment breaks (jaws cross teeth or lips).
# Williams Alligator uses SMMA (Smoothed Moving Average) with periods 13, 8, 5 and offsets 8, 5, 3.
# Designed for low trade frequency (target: 15-30/year) to minimize fee drag and improve generalization.
# Works in bull markets via bullish alignment and in bear markets via bearish alignment.
name = "6h_WilliamsAlligator_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - first value is SMA, then smoothed"""
    sma = np.full_like(data, np.nan, dtype=float)
    if len(data) < period:
        return sma
    # Initial SMA
    sma[period-1] = np.mean(data[:period])
    # Subsequent SMMA values
    for i in range(period, len(data)):
        sma[i] = (sma[i-1] * (period-1) + data[i]) / period
    return sma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3)
    # Using typical price (HLC/3) as input
    typical_price = (high + low + close) / 3.0
    
    jaw = smma(typical_price, 13)
    teeth = smma(typical_price, 8)
    lips = smma(typical_price, 5)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Bullish alignment: Jaw < Teeth < Lips
    # Bearish alignment: Jaw > Teeth > Lips
    bullish_alignment = (jaw < teeth) & (teeth < lips)
    bearish_alignment = (jaw > teeth) & (teeth > lips)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA50 on 12h close
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Sufficient warmup for indicators (max offset 8 + periods)
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alignment AND 12h EMA50 rising AND volume spike
            long_condition = bullish_alignment[i] and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Bearish alignment AND 12h EMA50 falling AND volume spike
            short_condition = bearish_alignment[i] and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Bullish alignment breaks
            if not bullish_alignment[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Bearish alignment breaks
            if not bearish_alignment[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals