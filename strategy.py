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
    
    # === Weekly data for market regime ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly EMA34 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Daily data for pivot points ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate standard pivot points: P, R1, S1, R2, S2
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl
    s1 = pivot - range_hl
    r2 = pivot + 2 * range_hl
    s2 = pivot - 2 * range_hl
    
    # Align daily pivot levels to 1d timeframe (no shift needed for daily)
    pivot_1d = pivot  # Already at daily frequency
    r1_1d = r1
    s1_1d = s1
    r2_1d = r2
    s2_1d = s2
    
    # Volume spike detection (20-period volume MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Daily ATR for volatility filter
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)),
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = high_1d[0] - low_1d[0]  # First value
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 50
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i]) or
            np.isnan(r2_1d[i]) or np.isnan(s2_1d[i]) or np.isnan(volume_spike[i]) or np.isnan(atr_1d[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        ema34_1w_val = ema34_1w_aligned[i]
        pivot_level = pivot_1d[i]
        r1_level = r1_1d[i]
        s1_level = s1_1d[i]
        r2_level = r2_1d[i]
        s2_level = s2_1d[i]
        vol_spike = volume_spike[i]
        atr = atr_1d[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price returns to pivot level (mean reversion)
            if price <= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price returns to pivot level (mean reversion)
            if price >= pivot_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, volatility filter, and weekly uptrend
            if price > r1_level and vol_spike and atr > 0 and price > ema34_1w_val:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, volatility filter, and weekly downtrend
            elif price < s1_level and vol_spike and atr > 0 and price < ema34_1w_val:
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

name = "1d_Pivot_R1_S1_Breakout_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0