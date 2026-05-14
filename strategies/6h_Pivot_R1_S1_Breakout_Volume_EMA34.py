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
    
    # === Daily data for pivot and ATR ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Pivot and R1/S1 levels (using close for pivot)
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    r1 = pivot + range_hl * 0.382
    s1 = pivot - range_hl * 0.382
    
    # === True Range and ATR (14-period) ===
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === 6h EMA for trend filter (34-period) ===
    ema_6h = pd.Series(close).ewm(span=34, min_periods=34, adjust=False).mean().values
    
    # Align HTF data to 6h timeframe
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    atr_14_6h = align_htf_to_ltf(prices, df_1d, atr_14)
    
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
        if (np.isnan(r1_6h[i]) or np.isnan(s1_6h[i]) or
            np.isnan(atr_14_6h[i]) or np.isnan(ema_6h[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        r1_level = r1_6h[i]
        s1_level = s1_6h[i]
        atr_val = atr_14_6h[i]
        ema_val = ema_6h[i]
        vol_spike = volume_spike[i]
        
        # === EXIT LOGIC ===
        if position == 1:  # Long position
            # Exit when price drops below S1 or volatility drops significantly
            if price < s1_level or (i > 0 and atr_val < atr_14_6h[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # Short position
            # Exit when price rises above R1 or volatility drops significantly
            if price > r1_level or (i > 0 and atr_val < atr_14_6h[i-1] * 0.7):
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # LONG: Price breaks above R1 with volume spike, above EMA34
            if price > r1_level and vol_spike and price > ema_val:
                signals[i] = 0.25
                position = 1
                continue
            
            # SHORT: Price breaks below S1 with volume spike, below EMA34
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

name = "6h_Pivot_R1_S1_Breakout_Volume_EMA34"
timeframe = "6h"
leverage = 1.0