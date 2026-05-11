#!/usr/bin/env python3
# 4h_Camarilla_Pivot_Bounce_1dTrend_Volume
# Hypothesis: Price reversals at Camarilla pivot levels (S3/S4 for long, R3/R4 for short)
# with 1-day trend confirmation and volume spike. Works in bull by buying S3/S4 dips in uptrend,
# and in bear by selling R3/R4 rallies in downtrend. Target: 20-40 trades/year.

name = "4h_Camarilla_Pivot_Bounce_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Calculate Camarilla levels from previous 1d bar ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas: H = high, L = low, C = close of previous day
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    R2 = close_1d + (high_1d - low_1d) * 1.1 / 6
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    R4 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    S2 = close_1d - (high_1d - low_1d) * 1.1 / 6
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    S4 = close_1d - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 4h (previous day's levels available at 00:00 UTC)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    R4_4h = align_htf_to_ltf(prices, df_1d, R4)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    S4_4h = align_htf_to_ltf(prices, df_1d, S4)
    
    # --- 1d trend: EMA34 ---
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # --- Volume spike: volume > 1.5 * 20-period average ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for calculations
    start_idx = max(20, 34)  # volume MA and EMA warmup
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price at or below S3 with volume spike in uptrend
            if low[i] <= S3_4h[i] and vol_spike[i] and close[i] > ema_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price at or above R3 with volume spike in downtrend
            elif high[i] >= R3_4h[i] and vol_spike[i] and close[i] < ema_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price crosses above S2 or trend changes
                if high[i] >= S2_4h[i] or close[i] < ema_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses below R2 or trend changes
                if low[i] <= R2_4h[i] or close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals