#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Camarilla_R1S1_Breakout_Volume_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # === Weekly: Camarilla pivot points ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # R1 = C + (H - L) * 1.1 / 12
    r1_1w = close_1w + (high_1w - low_1w) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    s1_1w = close_1w - (high_1w - low_1w) * 1.1 / 12
    # R2 = C + (H - L) * 1.1 / 6
    r2_1w = close_1w + (high_1w - low_1w) * 1.1 / 6
    # S2 = C - (H - L) * 1.1 / 6
    s2_1w = close_1w - (high_1w - low_1w) * 1.1 / 6
    # R3 = C + (H - L) * 1.1 / 4
    r3_1w = close_1w + (high_1w - low_1w) * 1.1 / 4
    # S3 = C - (H - L) * 1.1 / 4
    s3_1w = close_1w - (high_1w - low_1w) * 1.1 / 4
    # R4 = C + (H - L) * 1.1 / 2
    r4_1w = close_1w + (high_1w - low_1w) * 1.1 / 2
    # S4 = C - (H - L) * 1.1 / 2
    s4_1w = close_1w - (high_1w - low_1w) * 1.1 / 2
    
    # Align all weekly pivot levels
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    r2_1w_aligned = align_htf_to_ltf(prices, df_1w, r2_1w)
    s2_1w_aligned = align_htf_to_ltf(prices, df_1w, s2_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # === Daily: ATR(14) for volatility and stop loss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = np.abs(high[1:] - low[1:])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Daily: Volume condition ===
    vol_ma = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 50  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Get aligned values
        pivot = pivot_1w_aligned[i]
        r1 = r1_1w_aligned[i]
        s1 = s1_1w_aligned[i]
        r2 = r2_1w_aligned[i]
        s2 = s2_1w_aligned[i]
        r3 = r3_1w_aligned[i]
        s3 = s3_1w_aligned[i]
        r4 = r4_1w_aligned[i]
        s4 = s4_1w_aligned[i]
        current_atr = atr[i]
        current_close = prices['close'].iloc[i]
        current_volume = prices['volume'].iloc[i]
        current_vol_ma = vol_ma[i]
        
        # Skip if any value is NaN
        if (np.isnan(pivot) or np.isnan(r1) or np.isnan(s1) or np.isnan(r2) or np.isnan(s2) or
            np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4) or np.isnan(current_atr) or np.isnan(current_vol_ma)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 1.5x 20-day average volume
        vol_condition = current_volume > 1.5 * current_vol_ma
        
        if position == 0:
            # Long conditions:
            # 1. Price breaks above R1 with volume
            # 2. Price is below R2 (not overextended)
            if (current_close > r1 and
                vol_condition and
                current_close < r2):
                signals[i] = 0.25
                position = 1
                entry_price = current_close
            
            # Short conditions:
            # 1. Price breaks below S1 with volume
            # 2. Price is above S2 (not overextended)
            elif (current_close < s1 and
                  vol_condition and
                  current_close > s2):
                signals[i] = -0.25
                position = -1
                entry_price = current_close
        
        elif position == 1:
            # Long exit conditions:
            # 1. Price falls below S1 (strong support - take profit)
            # 2. ATR-based stop loss
            if (current_close <= s1 or
                current_close < entry_price - 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit conditions:
            # 1. Price rises above R1 (strong resistance - take profit)
            # 2. ATR-based stop loss
            if (current_close >= r1 or
                current_close > entry_price + 2.5 * current_atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals