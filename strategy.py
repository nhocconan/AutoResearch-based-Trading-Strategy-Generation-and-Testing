#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and trend filter.
# Uses daily Camarilla levels (S1/S2 for longs, R1/R2 for shorts) from prior day.
# Enters when price touches S1/S2/R1/R2 with volume spike and 4h EMA trend alignment.
# Works in bull/bear: buys dips in uptrend, sells rallies in downtrend.
# Target: 20-40 trades/year (~80-160 total over 4 years) to avoid fee drag.
name = "4h_Camarilla_Pivot_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Daily: Camarilla pivot levels (based on prior day's range) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC for today's Camarilla levels (no look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan  # first day has no prior
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels
    range_prev = prev_high - prev_low
    S1 = prev_close - (range_prev * 1.1 / 12)
    S2 = prev_close - (range_prev * 1.1 / 6)
    R1 = prev_close + (range_prev * 1.1 / 12)
    R2 = prev_close + (range_prev * 1.1 / 6)
    
    # Align to 4h (these levels are valid for the entire day after the daily bar closes)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    
    # === 4h: Trend filter (EMA34) ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # === 4h: Price and volume ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume ratio (current vs 20-period average)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Session filter: 08-20 UTC (align with major liquidity)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip outside session
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get values
        close_val = close[i]
        low_val = low[i]
        high_val = high[i]
        ema_val = ema34_4h_aligned[i]
        vol_ratio_val = vol_ratio[i]
        s1_val = S1_aligned[i]
        s2_val = S2_aligned[i]
        r1_val = R1_aligned[i]
        r2_val = R2_aligned[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_val) or np.isnan(vol_ratio_val) or 
            np.isnan(s1_val) or np.isnan(s2_val) or 
            np.isnan(r1_val) or np.isnan(r2_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price touches S1 or S2 in uptrend with volume spike
            long_condition = (
                ((low_val <= s1_val and close_val > s1_val) or  # Touched/broke S1
                 (low_val <= s2_val and close_val > s2_val)) and  # Touched/broke S2
                (close_val > ema_val) and                        # Uptrend filter
                (vol_ratio_val > 1.8)                            # Volume spike
            )
            # Short: Price touches R1 or R2 in downtrend with volume spike
            short_condition = (
                ((high_val >= r1_val and close_val < r1_val) or  # Touched/broke R1
                 (high_val >= r2_val and close_val < r2_val)) and  # Touched/broke R2
                (close_val < ema_val) and                        # Downtrend filter
                (vol_ratio_val > 1.8)                            # Volume spike
            )
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 (take profit) or breaks S2 (stop) or trend fails
            if (close_val >= r1_val or      # Profit target at R1
                close_val < s2_val or       # Stop if breaks S2
                close_val < ema_val):       # Trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S2 (take profit) or breaks R2 (stop) or trend fails
            if (close_val <= s2_val or      # Profit target at S2
                close_val > r2_val or       # Stop if breaks R2
                close_val > ema_val):       # Trend failure
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals