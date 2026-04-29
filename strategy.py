#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend with weekly pivot regime filter and volume confirmation
# Supertrend captures trend direction, weekly pivot defines market regime (bull/bear/range),
# volume spike confirms institutional participation. Works in bull via trend following,
# in bear via mean reversion at pivot extremes, and in range via pivot reversals.
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe

name = "6h_Supertrend_WeeklyPivot_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate weekly Supertrend (ATR=10, mult=3.0)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_1w + low_1w) / 2
    upperband = hl2 + 3.0 * atr_10
    lowerband = hl2 - 3.0 * atr_10
    
    supertrend = np.full_like(close_1w, np.nan, dtype=float)
    direction = np.full_like(close_1w, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_1w)):
        if np.isnan(supertrend[i-1]):
            # Initialize
            supertrend[i] = lowerband[i]
            direction[i] = 1
        else:
            if close_1w[i] > supertrend[i-1]:
                supertrend[i] = max(lowerband[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upperband[i], supertrend[i-1])
                direction[i] = -1
    
    # Align Supertrend direction to 6h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate weekly pivot points (using prior week's OHLC)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    
    # Shift OHLC by 1 to use prior week's data
    high_1w_shift = np.concatenate([[np.nan], high_1w[:-1]])
    low_1w_shift = np.concatenate([[np.nan], low_1w[:-1]])
    close_1w_shift = np.concatenate([[np.nan], close_1w[:-1]])
    
    pivot = (high_1w_shift + low_1w_shift + close_1w_shift) / 3
    r1 = 2 * pivot - low_1w_shift
    s1 = 2 * pivot - high_1w_shift
    r2 = pivot + (high_1w_shift - low_1w_shift)
    s2 = pivot - (high_1w_shift - low_1w_shift)
    r3 = high_1w_shift + 2 * (pivot - low_1w_shift)
    s3 = low_1w_shift - 2 * (high_1w_shift - pivot)
    
    # Align pivot levels to 6h
    pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or
            np.isnan(s2_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_dir = supertrend_dir_aligned[i]
        curr_pivot = pivot_aligned[i]
        curr_r1 = r1_aligned[i]
        curr_s1 = s1_aligned[i]
        curr_r2 = r2_aligned[i]
        curr_s2 = s2_aligned[i]
        curr_r3 = r3_aligned[i]
        curr_s3 = s3_aligned[i]
        curr_ema_1d = ema_34_1d_aligned[i]
        
        # Volume spike confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        # Handle exits
        if position == 1:  # Long position
            # Exit: Supertrend turns bearish OR price crosses below S1 (weakness)
            if curr_dir == -1 or curr_close < curr_s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Supertrend turns bullish OR price crosses above R1 (strength)
            if curr_dir == 1 or curr_close > curr_r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Regime determination based on weekly pivot
            # Bull regime: price above weekly pivot
            # Bear regime: price below weekly pivot
            # Range regime: price between S1 and R1
            
            # Long entry conditions:
            # 1. Supertrend bullish OR price near support in range
            # 2. Price above 1d EMA34 (bullish bias) OR at weekly S1/S2/S3 (mean reversion)
            # 3. Volume spike confirmation
            long_condition = False
            if curr_dir == 1:  # Supertrend bullish
                long_condition = curr_close > curr_ema_1d and vol_spike
            elif curr_s1 <= curr_close <= curr_r1:  # In range
                # Mean reversion at support levels
                near_support = (abs(curr_close - curr_s1) / curr_s1 < 0.005 or  # within 0.5% of S1
                              abs(curr_close - curr_s2) / curr_s2 < 0.01 or   # within 1% of S2
                              abs(curr_close - curr_s3) / curr_s3 < 0.015)    # within 1.5% of S3
                long_condition = near_support and vol_spike
            
            # Short entry conditions:
            # 1. Supertrend bearish OR price near resistance in range
            # 2. Price below 1d EMA34 (bearish bias) OR at weekly R1/R2/R3 (mean reversion)
            # 3. Volume spike confirmation
            short_condition = False
            if curr_dir == -1:  # Supertrend bearish
                short_condition = curr_close < curr_ema_1d and vol_spike
            elif curr_s1 <= curr_close <= curr_r1:  # In range
                # Mean reversion at resistance levels
                near_resistance = (abs(curr_close - curr_r1) / curr_r1 < 0.005 or  # within 0.5% of R1
                                 abs(curr_close - curr_r2) / curr_r2 < 0.01 or   # within 1% of R2
                                 abs(curr_close - curr_r3) / curr_r3 < 0.015)    # within 1.5% of R3
                short_condition = near_resistance and vol_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals