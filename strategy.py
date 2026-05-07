#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Williams Alligator with 1-day trend filter and 1-week volume confirmation.
# Long when: price > Alligator Jaw (13-period SMMA) AND 1-day EMA(34) rising AND 1-week volume > 1-week SMA(20) volume
# Short when: price < Alligator Teeth (8-period SMMA) AND 1-day EMA(34) falling AND 1-week volume > 1-week SMA(20) volume
# Exit when price crosses the Alligator Lips (5-period SMMA).
# Designed for 12h timeframe with low trade frequency (target: 12-37/year) to avoid fee drag.
# Uses Williams Alligator for trend/filter, 1d EMA for trend confirmation, 1w volume for conviction.
# Works in bull markets via price above Jaw in uptrend, in bear markets via price below Teeth in downtrend.
# Volume filter ensures participation from higher timeframe players.

name = "12h_WilliamsAlligator_1dEMA34_1wVolumeConfirm"
timeframe = "12h"
leverage = 1.0

def smma(source, length):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing"""
    return pd.Series(source).ewm(alpha=1/length, adjust=False).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator on 12h timeframe (using current timeframe data)
    # Jaw: 13-period SMMA of median price
    median_price = (high + low) / 2
    jaw = smma(median_price, 13)
    # Teeth: 8-period SMMA of median price
    teeth = smma(median_price, 8)
    # Lips: 5-period SMMA of median price
    lips = smma(median_price, 5)
    
    # 1-day EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # 1-week volume confirmation: volume > SMA(20) of volume
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    volume_sma_20_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    high_volume_1w = volume_1w > volume_sma_20_1w
    
    high_volume_aligned = align_htf_to_ltf(prices, df_1w, high_volume_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(high_volume_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > Jaw AND 1d EMA34 rising AND 1w high volume
            long_condition = (close[i] > jaw[i]) and ema_34_rising_aligned[i] and high_volume_aligned[i]
            # Short: price < Teeth AND 1d EMA34 falling AND 1w high volume
            short_condition = (close[i] < teeth[i]) and ema_34_falling_aligned[i] and high_volume_aligned[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Lips
            if close[i] < lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Lips
            if close[i] > lips[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals