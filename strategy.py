#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Williams Alligator with 1-day trend filter and volume confirmation.
# Long when: Price > Alligator's Jaw (13-period SMMA) AND Jaw > Teeth > Lips (bullish alignment) AND 1-day EMA50 rising AND volume > 1.5 * EMA20(volume).
# Short when: Price < Alligator's Jaw AND Jaw < Teeth < Lips (bearish alignment) AND 1-day EMA50 falling AND volume > 1.5 * EMA20(volume).
# Exit when price crosses back below/above the Alligator's Teeth (8-period SMMA).
# Designed for low trade frequency (target: 20-35/year) to minimize fee drift and improve generalization.
# Williams Alligator uses smoothed moving averages (SMMA) which reduce whipsaws in choppy markets.
# Works in bull markets via bullish alignment and upward price position, and in bear markets via bearish alignment and downward price position.
name = "4h_WilliamsAlligator_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_price) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) - all SMMA
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Bullish alignment: Lips > Teeth > Jaw
    bullish_alignment = (lips > teeth) & (teeth > jaw)
    # Bearish alignment: Lips < Teeth < Jaw
    bearish_alignment = (lips < teeth) & (teeth < jaw)
    
    # Exit condition: price crosses Teeth (8-period SMMA)
    
    # Load 1-day data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA50 on 1d close
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Rising if current > previous, falling if current < previous
    ema_50_rising = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1d, dtype=bool)
    ema_50_rising[1:] = ema_50_1d[1:] > ema_50_1d[:-1]
    ema_50_falling[1:] = ema_50_1d[1:] < ema_50_1d[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish alignment AND price > Jaw AND EMA50(1d) rising AND volume spike
            long_condition = bullish_alignment[i] and (close[i] > jaw[i]) and ema_50_rising_aligned[i] and volume_spike[i]
            # Short: Bearish alignment AND price < Jaw AND EMA50(1d) falling AND volume spike
            short_condition = bearish_alignment[i] and (close[i] < jaw[i]) and ema_50_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < Teeth (8-period SMMA)
            if close[i] < teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > Teeth (8-period SMMA)
            if close[i] > teeth[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals