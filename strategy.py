#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Regime
Hypothesis: Camarilla R1/S1 breakout with daily EMA34 trend filter, volume spike confirmation, and choppiness regime filter. 
In trending markets (CHOP < 38.2): trade breakouts in direction of daily EMA34. 
In ranging markets (CHOP > 61.8): fade extremes at R1/S1 with volume confirmation. 
Discrete sizing (0.25) to minimize fee drag. Target: 75-200 trades over 4 years.
Works in both bull and bear regimes by adapting to market structure via choppiness filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:  # Need 34 for daily EMA
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Need daily high, low, close - using 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: R1, S1, R2, S2, R3, S3, R4, S4
    # Formula: Close + (High - Low) * 1.1 / 12 for R1, etc.
    # We'll calculate these once per day and align to 4h
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla levels
    camarilla_r1 = daily_close + (daily_high - daily_low) * 1.1 / 12
    camarilla_s1 = daily_close - (daily_high - daily_low) * 1.1 / 12
    camarilla_r2 = daily_close + (daily_high - daily_low) * 1.1 / 6
    camarilla_s2 = daily_close - (daily_high - daily_low) * 1.1 / 6
    camarilla_r3 = daily_close + (daily_high - daily_low) * 1.1 / 4
    camarilla_s3 = daily_close - (daily_high - daily_low) * 1.1 / 4
    camarilla_r4 = daily_close + (daily_high - daily_low) * 1.1 / 2
    camarilla_s4 = daily_close - (daily_high - daily_low) * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Choppiness regime filter (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(N)
    tr1 = np.abs(high - low)
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_hl = max_high - min_low
    range_hl = np.where(range_hl == 0, 1e-10, range_hl)
    
    chop = 100 * np.log10(pd.Series(atr).rolling(window=14, min_periods=14).sum().values / range_hl) / np.log10(14)
    
    # Regime definitions
    chop_trending = chop < 38.2   # Strong trend
    chop_ranging = chop > 61.8    # Strong range
    chop_neutral = ~(chop_trending | chop_ranging)  # Weak trend/range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    bars_since_entry = 0
    
    # Start after warmup (need 34 for EMA, 14 for CHOP)
    start_idx = 34
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        close_val = close[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        r2_val = r2_aligned[i]
        s2_val = s2_aligned[i]
        ema_val = ema_34_aligned[i]
        chop_val = chop[i]
        vol_spike = volume_spike[i]
        
        # Regime-based logic
        if chop_trending:
            # Trending market: breakout in direction of daily EMA34
            long_condition = (close_val > r1_val and 
                            close_val > ema_val and 
                            vol_spike)
            short_condition = (close_val < s1_val and 
                             close_val < ema_val and 
                             vol_spike)
        elif chop_ranging:
            # Ranging market: fade extremes at R1/S1 with volume confirmation
            long_condition = (close_val < s1_val and 
                             vol_spike and 
                             close_val > s2_val)  # Not too deep
            short_condition = (close_val > r1_val and 
                             vol_spike and 
                             close_val < r2_val)  # Not too deep
        else:
            # Neutral regime: standard breakout with volume
            long_condition = (close_val > r1_val and 
                            vol_spike)
            short_condition = (close_val < s1_val and 
                             vol_spike)
        
        # Exit logic: opposite signal or volatility expansion
        exit_long = (close_val < s1_val) or (not vol_spike and position == 1)
        exit_short = (close_val > r1_val) or (not vol_spike and position == -1)
        
        # Minimum holding period: 2 bars
        if position != 0 and bars_since_entry < 2:
            # Hold position regardless of signals
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
            bars_since_entry = 0
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
            bars_since_entry = 0
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Regime"
timeframe = "4h"
leverage = 1.0