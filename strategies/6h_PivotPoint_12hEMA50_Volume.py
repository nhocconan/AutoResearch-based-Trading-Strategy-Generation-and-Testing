#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Pivot Point Reversal with 12h Trend Filter and Volume Confirmation
# Uses classic pivot points (PP, R1, S1, R2, S2) calculated from 12h OHLC.
# Enters long when price closes above R1 with 12h uptrend (close > EMA50) and volume > 1.5x average.
# Enters short when price closes below S1 with 12h downtrend (close < EMA50) and volume > 1.5x average.
# Exits when price returns to pivot point (PP) or reverses across EMA50.
# Pivot points provide objective support/resistance levels that work in all market regimes.
# Trend filter prevents counter-trend trades. Volume confirms institutional participation.
# Target: 60-120 total trades over 4 years (15-30/year) to minimize fee drag.

name = "6h_PivotPoint_12hEMA50_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data ONCE before loop for pivot points and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # === 12h Pivot Points (using typical OHLC) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Classic pivot point calculation
    pp = (high_12h + low_12h + close_12h) / 3.0
    r1 = 2 * pp - low_12h
    s1 = 2 * pp - high_12h
    r2 = pp + (high_12h - low_12h)
    s2 = pp - (high_12h - low_12h)
    
    # Align pivot levels to 6s timeframe
    pp_aligned = align_htf_to_ltf(prices, df_12h, pp)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # === 12h EMA50 for trend filter ===
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # === 6h Volume confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values  # 24 * 6h = 6 days
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(30, n):  # Start after warmup
        # Get values
        close_val = prices['close'].iloc[i]
        ema_val = ema_50_aligned[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(pp_val) or np.isnan(r1_val) or 
            np.isnan(s1_val) or np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price closes above R1, 12h uptrend, volume confirmation
            if close_val > r1_val and close_val > ema_val and vol_ratio_val > 1.5:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            # Short entry: price closes below S1, 12h downtrend, volume confirmation
            elif close_val < s1_val and close_val < ema_val and vol_ratio_val > 1.5:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
        
        elif position == 1:
            # Long exit: price returns to pivot point or trend breaks
            if close_val <= pp_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to pivot point or trend breaks
            if close_val >= pp_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals