#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 1d trend filter and volume confirmation.
# Long when price breaks above R3 AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when price breaks below S3 AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back inside Camarilla H-L range.
# This strategy targets volatility expansion phases with trend alignment to capture momentum moves
# while avoiding choppy markets. The 1d EMA34 filter ensures we trade with the higher timeframe trend.
# Volume confirmation ensures institutional participation and reduces false breakouts.
# Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d trend direction.

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Camarilla levels from previous day (using daily OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r3 = np.full_like(prev_close, np.nan)
    camarilla_s3 = np.full_like(prev_close, np.nan)
    camarilla_h = np.full_like(prev_close, np.nan)
    camarilla_l = np.full_like(prev_close, np.nan)
    
    valid = ~(np.isnan(prev_close) | np.isnan(prev_high) | np.isnan(prev_low))
    if np.any(valid):
        rng = prev_high[valid] - prev_low[valid]
        camarilla_r3[valid] = prev_close[valid] + rng * 1.1 / 6
        camarilla_s3[valid] = prev_close[valid] - rng * 1.1 / 6
        camarilla_h[valid] = prev_high[valid]
        camarilla_l[valid] = prev_low[valid]
    
    # Align Camarilla levels to 4h timeframe (wait for previous day to close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_h_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h)
    camarilla_l_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_h_aligned[i]) or np.isnan(camarilla_l_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 1d EMA34 rising, volume filter
            long_cond = (close[i] > camarilla_r3_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price breaks below S3, 1d EMA34 falling, volume filter
            short_cond = (close[i] < camarilla_s3_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back inside Camarilla H-L range (below H)
            if close[i] < camarilla_h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back inside Camarilla H-L range (above L)
            if close[i] > camarilla_l_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals