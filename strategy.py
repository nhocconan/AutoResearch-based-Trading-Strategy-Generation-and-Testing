#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA200 trend filter, volume spike confirmation, and session filter.
# Uses 1h primary timeframe targeting 15-37 trades/year (60-150 total over 4 years).
# 4h EMA200 provides strong trend filter for both bull and bear markets (price > EMA200 = bull, < EMA200 = bear).
# Camarilla H3/L3 from 4h provide institutional pivot points with proven edge.
# Volume spike (>2.0x 20-bar average) confirms breakout strength and filters low-momentum false breakouts.
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods.
# Position size 0.20 for capital preservation. Discrete levels minimize fee churn.

name = "1h_Camarilla_H3_L3_Breakout_4hEMA200_Trend_VolumeSpike_Session_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots and EMA200 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h Camarilla pivot levels (H3, L3)
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    range_4h = high_4h - low_4h
    h3_4h = close_4h + (high_4h - low_4h) * 1.1 / 4.0  # H3 = Close + 1.1*(Range)/4
    l3_4h = close_4h - (high_4h - low_4h) * 1.1 / 4.0  # L3 = Close - 1.1*(Range)/4
    
    # Calculate 4h EMA200 for trend filter
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 4h indicators to 1h timeframe
    h3_4h_aligned = align_htf_to_ltf(prices, df_4h, h3_4h)
    l3_4h_aligned = align_htf_to_ltf(prices, df_4h, l3_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # Calculate 1h volume spike: >2.0x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure sufficient history for EMA200
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_4h_aligned[i]) or
            np.isnan(l3_4h_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA200 direction
        price_above_ema = close[i] > ema_200_4h_aligned[i]
        price_below_ema = close[i] < ema_200_4h_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h3_4h_aligned[i]
        short_breakout = close[i] < l3_4h_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level (H4/L4 for tighter stop)
        # Calculate H4/L4 for exit
        h4_4h = close_4h + (high_4h - low_4h) * 1.1 / 2.0  # H4 = Close + 1.1*(Range)/2
        l4_4h = close_4h - (high_4h - low_4h) * 1.1 / 2.0  # L4 = Close - 1.1*(Range)/2
        h4_4h_aligned = align_htf_to_ltf(prices, df_4h, h4_4h)
        l4_4h_aligned = align_htf_to_ltf(prices, df_4h, l4_4h)
        
        long_exit = close[i] < l4_4h_aligned[i]  # Exit long at L4
        short_exit = close[i] > h4_4h_aligned[i]  # Exit short at H4
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals