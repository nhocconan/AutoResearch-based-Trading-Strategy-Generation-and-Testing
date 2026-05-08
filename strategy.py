#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Long when price touches or crosses above Camarilla S3 (support) AND 1d EMA34 rising AND volume > 1.3x 20-period average.
# Short when price touches or crosses below Camarilla R3 (resistance) AND 1d EMA34 falling AND volume > 1.3x 20-period average.
# Exit when price reaches the opposite Camarilla level (S1 for long, R1 for short) or reverses across the pivot point.
# Camarilla levels provide high-probability reversal zones in ranging markets, while EMA34 filter ensures alignment with higher timeframe trend.
# Volume confirmation filters out low-conviction moves. Target: 80-160 total trades over 4 years (20-40/year).

name = "4h_Camarilla_R3S3_Reversal_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d OHLC
    # Using previous day's close, high, low
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla formula: 
    #   H2 = C + (H-L)*1.1/6
    #   H3 = C + (H-L)*1.1/4
    #   H4 = C + (H-L)*1.1/2
    #   L3 = C - (H-L)*1.1/4
    #   L2 = C - (H-L)*1.1/6
    #   L1 = C - (H-L)*1.1/12
    #   P  = (H+L+C)/3
    # We'll use H4/L4 as outer bands and H3/L3 as entry, H1/L1 as exits
    rng = prev_high - prev_low
    camarilla_h4 = prev_close + rng * 1.1 / 2
    camarilla_l3 = prev_close - rng * 1.1 / 4
    camarilla_h3 = prev_close + rng * 1.1 / 4
    camarilla_l1 = prev_close - rng * 1.1 / 12
    camarilla_h1 = prev_close + rng * 1.1 / 12
    camarilla_p = (prev_high + prev_low + prev_close) / 3
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_p_aligned = align_htf_to_ltf(prices, df_1d, camarilla_p)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(camarilla_h1_aligned[i]) or np.isnan(camarilla_l1_aligned[i]) or 
            np.isnan(camarilla_p_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches or crosses above Camarilla L3 (support), 1d EMA34 rising, volume filter
            long_cond = (low[i] <= camarilla_l3_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price touches or crosses below Camarilla H3 (resistance), 1d EMA34 falling, volume filter
            short_cond = (high[i] >= camarilla_h3_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reaches Camarilla H1 (resistance) or crosses below pivot
            if high[i] >= camarilla_h1_aligned[i] or close[i] < camarilla_p_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches Camarilla L1 (support) or crosses above pivot
            if low[i] <= camarilla_l1_aligned[i] or close[i] > camarilla_p_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals