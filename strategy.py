#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band breakout with 1d trend filter and volume confirmation.
# Long when: close breaks above BB(20,2) upper band AND 1d EMA(34) rising AND volume > 1.5x 20-period average
# Short when: close breaks below BB(20,2) lower band AND 1d EMA(34) falling AND volume > 1.5x 20-period average
# Exit when price returns to BB middle band (20-period SMA).
# Designed for 6h timeframe to capture medium-term breakouts with trend alignment.
# Bollinger Bands provide dynamic support/resistance, 1d EMA filters for trend direction,
# volume confirmation avoids false breakouts. Works in both bull and bear markets by
# following the 1d trend direction for breakouts.

name = "6h_BollingerBreakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20, 2)
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    bb_upper = sma_20 + 2 * std_20
    bb_lower = sma_20 - 2 * std_20
    bb_middle = sma_20
    
    # Volume average (20-period)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d EMA(34) for trend filter
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
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Sufficient warmup for Bollinger Bands
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20[i]) or np.isnan(std_20[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close breaks above BB upper AND 1d EMA34 rising AND volume surge
            breakout_up = close[i] > bb_upper[i]
            vol_surge = volume[i] > 1.5 * vol_ma_20[i]
            
            if breakout_up and ema_34_rising_aligned[i] and vol_surge:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below BB lower AND 1d EMA34 falling AND volume surge
            elif close[i] < bb_lower[i] and ema_34_falling_aligned[i] and vol_surge:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns to BB middle band
            if close[i] <= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns to BB middle band
            if close[i] >= bb_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals