#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout (20) + 1d volume spike + 1w trend filter (EMA50)
# Enter long when price breaks above Donchian(20) high, 1d volume > 2.0x 20-bar average, and 1w EMA50 rising
# Enter short when price breaks below Donchian(20) low, 1d volume > 2.0x 20-bar average, and 1w EMA50 falling
# Exit when price crosses back below/above Donchian midpoint or volume fails
# Designed for low trade frequency (<30/year) with strong trend capture in both bull and bear markets

name = "12h_Donchian20_1dVolume_1wEMA50"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 12h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: 1d volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 2.0)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    ema50_1w_slope = ema50_1w[1:] - ema50_1w[:-1]
    ema50_1w_slope = np.concatenate([[0], ema50_1w_slope])
    ema50_1w_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for Donchian and EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(ema50_1w_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        highest_high_val = highest_high[i]
        lowest_low_val = lowest_low[i]
        donchian_mid_val = donchian_mid[i]
        vol_conf_val = vol_conf[i]
        ema50_1w_val = ema50_1w_aligned[i]
        ema50_1w_slope_val = ema50_1w_slope_aligned[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, volume confirmation, 1w uptrend
            if close[i] > highest_high_val and vol_conf_val and ema50_1w_slope_val > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, volume confirmation, 1w downtrend
            elif close[i] < lowest_low_val and vol_conf_val and ema50_1w_slope_val < 0:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below Donchian midpoint or volume fails
            if close[i] < donchian_mid_val or not vol_conf_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above Donchian midpoint or volume fails
            if close[i] > donchian_mid_val or not vol_conf_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals