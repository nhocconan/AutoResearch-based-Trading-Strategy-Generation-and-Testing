#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === Weekly data for pivot points (HTF) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Pivot, R1, S1 levels (standard formula)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_hl = high_1w - low_1w
    r1 = pivot + range_hl * 0.382
    s1 = pivot - range_hl * 0.382
    
    # === Daily EMA for trend filter (34-period) ===
    ema_daily = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF data to daily timeframe
    r1_daily = align_htf_to_ltf(prices, df_1w, r1)
    s1_daily = align_htf_to_ltf(prices, df_1w, s1)
    
    # === Volume spike detection (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or
            np.isnan(ema_daily[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_daily[i]
        s1_level = s1_daily[i]
        ema_val = ema_daily[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1
            if price < s1_level:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1
            if price > r1_level:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, above daily EMA34
            if price > r1_level and vol_spike and price > ema_val:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, below daily EMA34
            elif price < s1_level and vol_spike and price < ema_val:
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

name = "1d_Pivot_R1_S1_Breakout_Volume_EMA34Filter_1w"
timeframe = "1d"
leverage = 1.0