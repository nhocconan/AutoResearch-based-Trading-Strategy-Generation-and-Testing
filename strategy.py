#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla pivot breakout with 12h volume confirmation and 1d ATR volatility filter
    # Enter long on breakout above R4, short on breakdown below S4
    # Exit when price re-enters the Camarilla range (between H3 and L3)
    # Volume confirmation: >1.5x 20-bar average
    # Volatility filter: ATR(10) < ATR(30) on 1d (low volatility regime for cleaner breakouts)
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    # Camarilla levels provide structured support/resistance; breakouts with volume/vol filter reduce false signals
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data for Camarilla pivot calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate previous 6h bar's Camarilla levels
    # Use previous bar's high, low, close to avoid look-ahead
    prev_high = np.roll(high_6h, 1)
    prev_low = np.roll(low_6h, 1)
    prev_close = np.roll(close_6h, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + (range_ * 1.1 / 2)
    r3 = pivot + (range_ * 1.1 / 4)
    r2 = pivot + (range_ * 1.1 / 6)
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    s2 = pivot - (range_ * 1.1 / 6)
    s3 = pivot - (range_ * 1.1 / 4)
    s4 = pivot - (range_ * 1.1 / 2)
    
    # Align 6h Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_6h, r4)
    r3_aligned = align_htf_to_ltf(prices, df_6h, r3)
    r2_aligned = align_htf_to_ltf(prices, df_6h, r2)
    r1_aligned = align_htf_to_ltf(prices, df_6h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_6h, s1)
    s2_aligned = align_htf_to_ltf(prices, df_6h, s2)
    s3_aligned = align_htf_to_ltf(prices, df_6h, s3)
    s4_aligned = align_htf_to_ltf(prices, df_6h, s4)
    h3_aligned = r3_aligned  # H3 is same as R3
    l3_aligned = s3_aligned  # L3 is same as S3
    
    # Get 12h data for volume confirmation (HTF)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    avg_volume_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_12h)
    avg_volume_12h_aligned = align_htf_to_ltf(prices, df_12h, avg_volume_12h)
    volume_confirmed = volume_12h_aligned > (1.5 * avg_volume_12h_aligned)
    
    # Get 1d data for ATR-based volatility filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range for 1d
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR(10) and ATR(30) for 1d
    atr_10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr_30_1d = pd.Series(tr).ewm(span=30, adjust=False, min_periods=30).mean().values
    
    # Align 1d ATR values to 6h timeframe
    atr_10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_10_1d)
    atr_30_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_30_1d)
    
    # Volatility filter: ATR(10) < ATR(30) (low volatility regime)
    vol_filter = atr_10_1d_aligned < atr_30_1d_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(2, n):  # Start from 2 to ensure we have previous bar data
        # Skip if data not ready
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_confirmed[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions (using current bar's close vs current bar's levels)
        breakout_up = close[i] > r4_aligned[i]  # break above R4
        breakout_down = close[i] < s4_aligned[i]  # break below S4
        
        # Re-entry conditions (price back inside H3-L3 range)
        reentry_long = (position == 1 and close[i] > h3_aligned[i])
        reentry_short = (position == -1 and close[i] < l3_aligned[i])
        
        # Entry conditions with volume confirmation and volatility filter
        long_entry = breakout_up and volume_confirmed[i] and vol_filter[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and vol_filter[i] and position != -1
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif reentry_long or reentry_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_1d_camarilla_breakout_vol_filter_v1"
timeframe = "6h"
leverage = 1.0