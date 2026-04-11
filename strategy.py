#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_camarilla_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return signals
    
    # Calculate weekly EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's range
    range_1d = high_1d - low_1d
    
    # Camarilla levels (based on previous close)
    l4 = close_1d - (range_1d * 1.1000)
    l3 = close_1d - (range_1d * 1.1000 / 2)
    h3 = close_1d + (range_1d * 1.1000 / 2)
    h4 = close_1d + (range_1d * 1.1000)
    
    # Shift by 1 to use only completed 1d bars
    l4 = np.roll(l4, 1)
    l3 = np.roll(l3, 1)
    h3 = np.roll(h3, 1)
    h4 = np.roll(h4, 1)
    l4[0] = np.nan
    l3[0] = np.nan
    h3[0] = np.nan
    h4[0] = np.nan
    
    # Align 1d Camarilla levels to 12h timeframe
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    
    # Volume confirmation: volume > 2.0x 50-period average on 12h
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if (np.isnan(l4_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(h4_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_50[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        price_high = high[i]
        price_low = low[i]
        volume_current = volume[i]
        vol_ma = vol_ma_50[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 2.0 * vol_ma
        
        # Weekly trend filter
        weekly_uptrend = price_close > ema_50_1w_aligned[i]
        weekly_downtrend = price_close < ema_50_1w_aligned[i]
        
        # Long: price breaks above H3/H4 with volume in weekly uptrend
        long_signal = volume_confirmed and weekly_uptrend and (price_high > h3_aligned[i] or price_high > h4_aligned[i])
        
        # Short: price breaks below L3/L4 with volume in weekly downtrend
        short_signal = volume_confirmed and weekly_downtrend and (price_low < l3_aligned[i] or price_low < l4_aligned[i])
        
        # Exit when price returns to the previous day's close (pivot point)
        prev_close_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
        if np.isnan(prev_close_aligned[i]):
            pivot_value = price_close
        else:
            pivot_value = prev_close_aligned[i]
        
        exit_long = position == 1 and price_close < pivot_value
        exit_short = position == -1 and price_close > pivot_value
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            # Maintain current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Hypothesis: Camarilla pivot breakout strategy on 12h timeframe with weekly trend filter.
# Uses 1d Camarilla levels (L3, L4, H3, H4) from the previous day's price action.
# Enters long when price breaks above H3 or H4 with volume confirmation (>2x average volume) during weekly uptrend (price > weekly 50 EMA).
# Enters short when price breaks below L3 or L4 with volume confirmation during weekly downtrend (price < weekly 50 EMA).
# Exits when price returns to the previous day's close (pivot point).
# The Camarilla levels identify key support/resistance levels where price often reverses or accelerates.
# Weekly EMA(50) filter ensures we trade with the higher timeframe momentum, reducing whipsaw.
# Volume confirmation (>2x average) reduces false breakouts.
# Designed for low trade frequency (target: 15-35 trades/year) to minimize fee drag on 12h timeframe.
# Works in both bull and bear markets by trading breakouts in the direction of the weekly trend.