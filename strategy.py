#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for pivot points and ADX ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points: P, R1, S1
    pivot_1w = (high_1w + low_1w + close_1w) / 3
    range_hl_1w = high_1w - low_1w
    r1_1w = pivot_1w + range_hl_1w
    s1_1w = pivot_1w - range_hl_1w
    
    # Calculate weekly ADX (14-period) for trend strength
    tr_1w = np.maximum(high_1w - low_1w,
                       np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                  np.abs(low_1w - np.roll(close_1w, 1))))
    tr_1w[0] = high_1w[0] - low_1w[0]
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]),
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]),
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    tr14 = pd.Series(tr_1w).rolling(window=14, min_periods=14).sum().values
    plus_di14 = 100 * (pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / tr14)
    minus_di14 = 100 * (pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / tr14)
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14 + 1e-10)
    adx_1w = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align weekly data to daily timeframe
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1_1w)
    adx_1d = align_htf_to_ltf(prices, df_1w, adx_1w)
    
    # Daily ADX for additional trend filter
    tr_1d = np.maximum(high - low,
                       np.maximum(np.abs(high - np.roll(close, 1)),
                                  np.abs(low - np.roll(close, 1))))
    tr_1d[0] = high[0] - low[0]
    plus_dm_1d = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]),
                          np.maximum(high[1:] - high[:-1], 0), 0)
    plus_dm_1d = np.insert(plus_dm_1d, 0, 0)
    minus_dm_1d = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]),
                           np.maximum(low[:-1] - low[1:], 0), 0)
    minus_dm_1d = np.insert(minus_dm_1d, 0, 0)
    tr14_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    plus_di14_1d = 100 * (pd.Series(plus_dm_1d).rolling(window=14, min_periods=14).sum().values / tr14_1d)
    minus_di14_1d = 100 * (pd.Series(minus_dm_1d).rolling(window=14, min_periods=14).sum().values / tr14_1d)
    dx_1d = 100 * np.abs(plus_di14_1d - minus_di14_1d) / (plus_di14_1d + minus_di14_1d + 1e-10)
    adx_1d_local = pd.Series(dx_1d).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or
            np.isnan(adx_1d[i]) or np.isnan(adx_1d_local[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        pivot_level = pivot_1d[i]
        r1_level = r1_1d[i]
        s1_level = s1_1d[i]
        adx_weekly = adx_1d[i]
        adx_daily = adx_1d_local[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to weekly pivot level (mean reversion)
            if price <= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to weekly pivot level (mean reversion)
            if price >= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike and strong weekly trend (ADX > 25)
            if price > r1_level and vol_spike and adx_weekly > 25 and adx_daily > 20:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike and strong weekly trend (ADX > 25)
            elif price < s1_level and vol_spike and adx_weekly > 25 and adx_daily > 20:
                signals[i] = -0.25
                position = -1
                continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "1d_Pivot_R1_S1_Breakout_Volume_ADXFilter"
timeframe = "1d"
leverage = 1.0