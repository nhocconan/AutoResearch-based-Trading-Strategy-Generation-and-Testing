#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 12h primary timeframe to reduce trade frequency while capturing medium-term trends.
# Camarilla pivot levels provide strong support/resistance structure, filtered by 1d EMA34 trend
# and volume spikes to avoid false breakouts. Works in both bull and bear markets by
# following the 1d trend while using Camarilla levels as structure.
# Target: 50-150 total trades over 4 years (12-37/year). Size: 0.25.

name = "12h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours (08-20 UTC) to avoid TypeError
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Get 1d data for Camarilla pivot calculation (using previous day's OHLC)
    df_1d_for_pivot = get_htf_data(prices, '1d')
    if len(df_1d_for_pivot) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d_for_pivot['high'].values
    low_1d = df_1d_for_pivot['low'].values
    close_1d = df_1d_for_pivot['close'].values
    
    # Camarilla levels: R3, R4, S3, S4
    # R3 = close + (high - low) * 1.1/4
    # S3 = close - (high - low) * 1.1/4
    # R4 = close + (high - low) * 1.1/2
    # S4 = close - (high - low) * 1.1/2
    pivot_range = high_1d - low_1d
    r3 = close_1d + pivot_range * 1.1 / 4
    s3 = close_1d - pivot_range * 1.1 / 4
    r4 = close_1d + pivot_range * 1.1 / 2
    s4 = close_1d - pivot_range * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe (using previous day's close for calculation)
    r3_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d_for_pivot, s4)
    
    # 12h volume spike: >1.5x 20-bar average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # EMA34 needs 34 bars, volume MA needs 20, plus buffer
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or
            np.isnan(s4_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Skip outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: 1d EMA34 direction
        price_above_ema = close[i] > ema_34_1d_aligned[i]
        price_below_ema = close[i] < ema_34_1d_aligned[i]
        
        # Camarilla breakout conditions
        long_breakout = close[i] > r3_aligned[i]  # Break above R3
        short_breakout = close[i] < s3_aligned[i]  # Break below S3
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        long_entry = price_above_ema and long_breakout and vol_confirm
        short_entry = price_below_ema and short_breakout and vol_confirm
        
        # Exit conditions: opposite Camarilla level (S3 for long, R3 for short)
        long_exit = close[i] < s3_aligned[i]
        short_exit = close[i] > r3_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals