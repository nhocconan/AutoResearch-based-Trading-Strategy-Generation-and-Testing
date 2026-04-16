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
    
    # === 1d Pivot Points (Camarilla) ===
    df_1d = get_htf_data(prices, '1d')
    # Pivot points from previous day
    pp = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3
    r1 = pp + (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    s1 = pp - (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    
    # Align pivot levels to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === Volume Confirmation (1d volume spike) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    # === 1w Trend Filter: EMA34 on weekly close ===
    df_1w = get_htf_data(prices, '1w')
    ema_34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean()
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w.values)
    
    signals = np.zeros(n)
    
    # Warmup: EMA34 and pivot calculation require ~34 bars
    warmup = 40
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_34_1w_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema34_1w = ema_34_1w_aligned[i]
        
        # === EXIT LOGIC: Close position when price returns to pivot zone ===
        if position == 1:  # Long position
            # Exit when price crosses back below R1 (failed breakout)
            if price < r1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above S1 (failed breakdown)
            if price > s1_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume confirmation and price > weekly EMA34 (bullish bias)
            if price > r1_aligned[i] and vol_spike and price > ema34_1w:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume confirmation and price < weekly EMA34 (bearish bias)
            elif price < s1_aligned[i] and vol_spike and price < ema34_1w:
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

name = "1d_Pivot_R1_S1_Breakout_Volume_WeeklyEMA34Filter"
timeframe = "1d"
leverage = 1.0