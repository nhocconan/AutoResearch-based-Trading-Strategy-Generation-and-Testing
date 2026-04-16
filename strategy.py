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
    
    # === 1d Pivot Points (Weekly-based: using previous week's close) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot: use previous week's OHLC (weekly bar aligned to daily)
    # For simplicity, we use previous day's OHLC but note: we could use weekly if data available
    # Using previous day's OHLC for pivot (standard daily pivot)
    pp = (df_1d['high'].shift(1) + df_1d['low'].shift(1) + df_1d['close'].shift(1)) / 3
    r1 = pp + (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    s1 = pp - (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 6
    r2 = pp + (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 4
    s2 = pp - (df_1d['high'].shift(1) - df_1d['low'].shift(1)) * 1.1 / 4
    
    # Align pivot levels to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    
    # === Volume Confirmation (20-period volume MA) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)  # Strong volume spike
    
    # === 60-period EMA Trend Filter (on 6h) ===
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators have valid data
    warmup = 60  # Need EMA60 and data alignment
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(ema_60[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        ema60 = ema_60[i]
        
        # === EXIT LOGIC: Exit when price returns to pivot zone (PP) ===
        if position == 1:  # Long position
            # Exit when price crosses back below pivot point (failed bullish continuation)
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price crosses back above pivot point (failed bearish continuation)
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R2 with volume confirmation and price > EMA60 (strong bullish)
            if price > r2_aligned[i] and vol_spike and price > ema60:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S2 with volume confirmation and price < EMA60 (strong bearish)
            elif price < s2_aligned[i] and vol_spike and price < ema60:
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

name = "6h_Pivot_R2_S2_Breakout_Volume_EMA60Filter"
timeframe = "6h"
leverage = 1.0