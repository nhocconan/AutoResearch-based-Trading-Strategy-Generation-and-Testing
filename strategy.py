#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Weekly Camarilla Pivot breakout with 1-week trend filter (EMA200) and volume confirmation
# Long when price breaks above Camarilla H3 level AND price > weekly EMA200 AND volume > 2x 20-period average
# Short when price breaks below Camarilla L3 level AND price < weekly EMA200 AND volume > 2x 20-period average
# Exit when price crosses back inside the Camarilla H3-L3 range
# Uses weekly timeframe for structure, daily for entry timing
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while capturing major moves

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for Camarilla pivots and EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Camarilla pivot levels
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + (Range * 1.1)
    # L3 = Pivot - (Range * 1.1)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    camarilla_h3 = pivot + (weekly_range * 1.1)
    camarilla_l3 = pivot - (weekly_range * 1.1)
    
    # Align weekly Camarilla levels to daily timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l3)
    
    # Calculate weekly EMA200 for trend filter
    ema200_1w = pd.Series(weekly_close).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations (200 for EMA200 + buffer)
    start = 220
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0
        
        if position == 0:
            # Long setup: breakout above Camarilla H3 + above weekly EMA200 + volume confirmation
            if (price > camarilla_h3_aligned[i] and price > ema200_1w_aligned[i] and vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: breakdown below Camarilla L3 + below weekly EMA200 + volume confirmation
            elif (price < camarilla_l3_aligned[i] and price < ema200_1w_aligned[i] and vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls back below Camarilla L3 (opposite side)
            if price < camarilla_l3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price rises back above Camarilla H3 (opposite side)
            if price > camarilla_h3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_WeeklyCamarilla_EMA200_Volume"
timeframe = "1d"
leverage = 1.0