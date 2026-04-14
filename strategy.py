#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Camarilla Pivot Level Breakout with 1-day Volume Spike and 1-day Choppiness Regime Filter
# Long when price breaks above H4 (resistance) with volume > 1.5x 24-period average and CHOP > 61.8 (range)
# Short when price breaks below L4 (support) with volume > 1.5x 24-period average and CHOP > 61.8
# Exit when price returns to the Pivot Point (midpoint)
# Uses Camarilla levels for precise support/resistance, volume for breakout confirmation, CHOP to avoid trending markets
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from previous day's OHLC
    # H4 = C + 1.1*(H-L), L4 = C - 1.1*(H-L)
    # H3 = C + 1.1*(H-L)/2, L3 = C - 1.1*(H-L)/2
    # H2 = C + 1.1*(H-L)/4, L2 = C - 1.1*(H-L)/4
    # H1 = C + 1.1*(H-L)/6, L1 = C - 1.1*(H-L)/6
    # Pivot = (H+L+C)/3
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    hl_range = prev_high - prev_low
    H4 = prev_close + 1.1 * hl_range
    L4 = prev_close - 1.1 * hl_range
    Pivot = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    Pivot_aligned = align_htf_to_ltf(prices, df_1d, Pivot)
    
    # Calculate volume average for confirmation (24-period = 6 hours)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate Choppiness Index on 1d for regime filter
    # CHOP = 100 * LOG10(SUM(ATR(14)) / (N * LOG10(N))) / LOG10(N)
    # Simplified: CHOP = 100 * LOG10(ATR_sum) / LOG10(N) where ATR_sum = sum of true range over N periods
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(low_1d[1:] - high_1d[:-1], np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.maximum(tr1, tr2)
    tr = np.concatenate([[np.nan], tr])  # align with original index
    atr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    n_periods = 14
    chop = 100 * np.log10(pd.Series(atr14).rolling(window=n_periods, min_periods=n_periods).sum().values) / np.log10(n_periods)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=0)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(Pivot_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: break above H4 with volume confirmation and range regime
            if (price > H4_aligned[i] and vol > vol_threshold and chop_aligned[i] > 61.8):
                position = 1
                signals[i] = position_size
            # Short setup: break below L4 with volume confirmation and range regime
            elif (price < L4_aligned[i] and vol > vol_threshold and chop_aligned[i] > 61.8):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to Pivot Point
            if price <= Pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: return to Pivot Point
            if price >= Pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Camarilla_H4L4_Volume_CHOP"
timeframe = "4h"
leverage = 1.0