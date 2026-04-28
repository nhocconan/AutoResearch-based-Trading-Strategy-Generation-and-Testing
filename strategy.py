#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla H3/L3 breakout with 4h trend filter (EMA50), volume spike confirmation, and session filter (08-20 UTC).
# Uses 1h primary timeframe targeting 15-30 trades/year (60-120 total over 4 years).
# 4h EMA50 provides medium-term trend filter: bull when price > EMA50, bear when price < EMA50.
# Camarilla H3/L3 from 1d provide institutional pivot points with proven edge in both bull and bear markets.
# Volume spike (>2.0x 24-bar average) confirms breakout strength and filters low-momentum false breakouts.
# Session filter reduces noise trades during low-liquidity periods.
# Position size 0.20 for capital preservation. Discrete levels minimize fee churn.

name = "1h_Camarilla_H3_L3_Breakout_4hEMA50_Trend_VolumeSpike_Session_v1"
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
    
    # Get 1d data for Camarilla pivots (H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = close_1d + (high_1d - low_1d) * 1.1 / 4.0  # H3 = Close + 1.1*(Range)/4
    l3_1d = close_1d - (high_1d - low_1d) * 1.1 / 4.0  # L3 = Close - 1.1*(Range)/4
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h volume spike: >2.0x 24-bar average volume (to account for session gaps)
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient history for EMA50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 4h EMA50 direction (price above/below EMA50)
        price_above_ema = close[i] > ema_50_4h_aligned[i]
        price_below_ema = close[i] < ema_50_4h_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > h3_1d_aligned[i]
        short_breakout = close[i] < l3_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level (H4/L4 for tighter stop)
        # Calculate H4/L4 for exit
        h4_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0  # H4 = Close + 1.1*(Range)/2
        l4_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0  # L4 = Close - 1.1*(Range)/2
        h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
        l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
        
        long_exit = close[i] < l4_1d_aligned[i]  # Exit long at L4
        short_exit = close[i] > h4_1d_aligned[i]  # Exit short at H4
        
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