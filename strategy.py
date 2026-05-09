#!/usr/bin/env python3
# Hypothesis: 4h Bollinger Band squeeze breakout with 1d trend filter (EMA34) and volume confirmation
# Long when Bollinger Band width < 20th percentile (squeeze), price breaks above upper band, EMA34 > EMA34 previous, volume > 1.5x avg
# Short when Bollinger Band width < 20th percentile (squeeze), price breaks below lower band, EMA34 < EMA34 previous, volume > 1.5x avg
# Exit when price crosses back inside Bollinger Bands or Bollinger Band width > 80th percentile (expansion)
# Designed to capture volatility breakouts in both trending and ranging markets
# Target: 100-200 total trades over 4 years (25-50/year) with size 0.25

name = "4h_BB_Squeeze_Breakout_EMA34_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Bollinger Bands (20, 2) on 4h data
    bb_length = 20
    bb_mult = 2.0
    basis = pd.Series(close).rolling(window=bb_length, min_periods=bb_length).mean().values
    dev = bb_mult * pd.Series(close).rolling(window=bb_length, min_periods=bb_length).std().values
    upper = basis + dev
    lower = basis - dev
    bb_width = upper - lower
    
    # Calculate Bollinger Band width percentiles for squeeze/expansion detection
    bb_width_series = pd.Series(bb_width)
    bb_width_pct = bb_width_series.rolling(window=50, min_periods=50).rank(pct=True).values * 100
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:  # Need enough data for EMA34
        return np.zeros(n)
    
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_prev = np.roll(ema_34_1d, 1)
    ema_34_1d_prev[0] = ema_34_1d[0]  # Handle first value
    ema_34_1d_rising = ema_34_1d > ema_34_1d_prev
    ema_34_1d_falling = ema_34_1d < ema_34_1d_prev
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_rising)
    ema_34_1d_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bb_width_pct[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_34_1d_rising_aligned[i]) or 
            np.isnan(ema_34_1d_falling_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Bollinger squeeze (width < 20th percentile), price breaks above upper band, EMA rising, volume confirmation
            if (bb_width_pct[i] < 20 and 
                close[i] > upper[i] and 
                ema_34_1d_rising_aligned[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: Bollinger squeeze (width < 20th percentile), price breaks below lower band, EMA falling, volume confirmation
            elif (bb_width_pct[i] < 20 and 
                  close[i] < lower[i] and 
                  ema_34_1d_falling_aligned[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses back inside Bollinger Bands or Bollinger Band expansion (width > 80th percentile)
            if (close[i] < basis[i]) or (bb_width_pct[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses back inside Bollinger Bands or Bollinger Band expansion (width > 80th percentile)
            if (close[i] > basis[i]) or (bb_width_pct[i] > 80):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals