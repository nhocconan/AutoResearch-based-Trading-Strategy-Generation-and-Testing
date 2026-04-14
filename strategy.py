#/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Calculate resistance levels
    R1 = close_1d + range_1d * 1.1 / 12
    R2 = close_1d + range_1d * 1.1 / 6
    R3 = close_1d + range_1d * 1.1 / 4
    R4 = close_1d + range_1d * 1.1 / 2
    
    # Calculate support levels
    S1 = close_1d - range_1d * 1.1 / 12
    S2 = close_1d - range_1d * 1.1 / 6
    S3 = close_1d - range_1d * 1.1 / 4
    S4 = close_1d - range_1d * 1.1 / 2
    
    # Align pivot levels to 4h timeframe (previous day's levels available after daily close)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Trend filter: 50-period EMA on 4h
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 20)  # 50 for EMA50, 20 for volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(R1_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(ema50[i]) or 
            np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long setup: price touches or breaks above S1 with volume confirmation and above EMA50
            if price >= S1_aligned[i] and price <= S1_aligned[i] * 1.005 and vol > 1.5 * avg_vol[i] and price > ema50[i]:
                position = 1
                signals[i] = position_size
            # Short setup: price touches or breaks below R1 with volume confirmation and below EMA50
            elif price <= R1_aligned[i] and price >= R1_aligned[i] * 0.995 and vol > 1.5 * avg_vol[i] and price < ema50[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R2 (profit target) or breaks below S1 (stop loss)
            if price >= R2_aligned[i] or price < S1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S2 (profit target) or breaks above R1 (stop loss)
            if price <= S2_aligned[i] or price > R1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Touch_Volume"
timeframe = "4h"
leverage = 1.0