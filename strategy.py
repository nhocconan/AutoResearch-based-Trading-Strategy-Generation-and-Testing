#!/usr/bin/env python3
"""
1h_Pivot_R1_S1_Breakout_Volume_Filter_v1
1h strategy using daily Camarilla pivot points (R1/S1) breakouts with volume spike and ATR filter.
Breakout above R1 with volume > 1.5x average and ATR > 0.5% of price -> long.
Breakdown below S1 with volume > 1.5x average and ATR > 0.5% of price -> short.
Exit when price returns to pivot point (PP).
Uses 4h EMA50 for trend alignment.
Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # === Daily Pivot Points (using prior day's data) ===
    # We'll calculate pivots from daily OHLC, but need to be careful about timing
    # Use daily data from 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate typical price for pivot
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Pivot point
    pp = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Resistance and support levels
    r1 = 2 * pp - df_1d['low']
    s1 = 2 * pp - df_1d['high']
    r2 = pp + (df_1d['high'] - df_1d['low'])
    s2 = pp - (df_1d['high'] - df_1d['low'])
    
    # Align to 1h timeframe (will use previous day's values)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    
    # === ATR(14) for volatility filter ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # === Volume average (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 4h EMA50 for trend filter ===
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 100
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(pp_aligned[i]) or 
            np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_4h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Calculate volume spike condition
        vol_spike = volume[i] > 1.5 * vol_ma[i]
        
        # Calculate ATR filter (ATR > 0.5% of price)
        atr_filter = atr[i] > 0.005 * close[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above R1, volume spike, ATR filter, price above 4h EMA50
            if (close[i] > r1_aligned[i] and 
                vol_spike and 
                atr_filter and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
                continue
            # Short: price breaks below S1, volume spike, ATR filter, price below 4h EMA50
            elif (close[i] < s1_aligned[i] and 
                  vol_spike and 
                  atr_filter and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
                continue
        
        # Exit logic: return to pivot point (PP)
        elif position == 1:
            # Exit long: price crosses below PP
            if close[i] < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above PP
            if close[i] > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Pivot_R1_S1_Breakout_Volume_Filter_v1"
timeframe = "1h"
leverage = 1.0