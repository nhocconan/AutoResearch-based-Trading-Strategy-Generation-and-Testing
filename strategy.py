#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_Camarilla_Pivot_Volume_Trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 1 or len(df_1w) < 1:
        return np.zeros(n)
    
    # === 1d: Calculate Camarilla pivot levels (standard) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla pivot calculations
    range_1d = high_1d - low_1d
    close_prev = close_1d  # Using same period close for calculation
    
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1_1d = close_1d + range_1d * 1.1 / 12
    s1_1d = close_1d - range_1d * 1.1 / 12
    # R2 = C + (H-L)*1.1/6, S2 = C - (H-L)*1.1/6
    r2_1d = close_1d + range_1d * 1.1 / 6
    s2_1d = close_1d - range_1d * 1.1 / 6
    # R3 = C + (H-L)*1.1/4, S3 = C - (H-L)*1.1/4
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    # R4 = C + (H-L)*1.1/2, S4 = C - (H-L)*1.1/2
    r4_1d = close_1d + range_1d * 1.1 / 2
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align all Camarilla levels
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    r2_1d_aligned = align_htf_to_ltf(prices, df_1d, r2_1d)
    s2_1d_aligned = align_htf_to_ltf(prices, df_1d, s2_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1w: EMA50 for trend direction ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 12h: ATR(14) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        ema_trend = ema_50_1w_aligned[i]
        pivot = pivot_1d_aligned[i]
        r1 = r1_1d_aligned[i]
        s1 = s1_1d_aligned[i]
        r2 = r2_1d_aligned[i]
        s2 = s2_1d_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        
        # Skip if any value is NaN
        if (np.isnan(ema_trend) or np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or
            np.isnan(r2) or np.isnan(s2) or np.isnan(r3) or np.isnan(s3) or
            np.isnan(r4) or np.isnan(s4) or np.isnan(current_atr)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # === Volume condition: current volume > 1.5x 20-period 12h average volume ===
        if i >= 20:
            vol_ma = np.mean(prices['volume'].iloc[i-20:i].values)
            vol_condition = current_volume > 1.5 * vol_ma
        else:
            vol_condition = False
        
        if position == 0:
            # Long conditions:
            # 1. Price above weekly EMA50 (uptrend)
            # 2. Price breaks above S3 with volume (strong support break)
            # 3. Price is below R3 (not overextended)
            if (current_close > ema_trend and
                current_close > s3 and
                vol_condition and
                current_close < r3):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price below weekly EMA50 (downtrend)
            # 2. Price breaks below R3 with volume (strong resistance break)
            # 3. Price is above S3 (not overextended)
            elif (current_close < ema_trend and
                  current_close < r3 and
                  vol_condition and
                  current_close > s3):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below weekly EMA50 (trend change)
            # 2. Price hits R3 (strong resistance - take profit)
            # 3. ATR-based stop loss
            if (current_close < ema_trend or
                current_close >= r3 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above weekly EMA50 (trend change)
            # 2. Price hits S3 (strong support - take profit)
            # 3. ATR-based stop loss
            if (current_close > ema_trend or
                current_close <= s3 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals