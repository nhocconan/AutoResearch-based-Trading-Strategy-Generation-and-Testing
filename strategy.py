#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot long at L3 with volume confirmation and weekly trend filter
# Long when price touches L3 (Camarilla support) AND volume > 1.5x average AND weekly EMA200 trending up
# Exit when price reaches H3 (Camarilla resistance) or closes below L3
# Camarilla levels provide precise intraday support/resistance, volume confirms institutional interest,
# weekly trend ensures alignment with higher timeframe momentum. Designed for 12h timeframe to balance opportunity and cost.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean()
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: H4 = close + 1.1*(high-low)/2, H3 = close + 1.1*(high-low)/4, L3 = close - 1.1*(high-low)/4
    # We'll use previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # Previous day high
    prev_low = df_1d['low'].shift(1).values    # Previous day low
    prev_close = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate Camarilla levels
    rng = prev_high - prev_low
    L3 = prev_close - 1.1 * rng / 4
    H3 = prev_close + 1.1 * rng / 4
    
    # Align Camarilla levels to 12h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    
    # Calculate volume average for confirmation (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # Need enough data for weekly EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 1.5
        
        if position == 0:
            # Long setup: price touches L3 (within 0.5%) AND volume confirmation AND weekly uptrend
            if (abs(close_val - L3_aligned[i]) / L3_aligned[i] < 0.005 and 
                vol > vol_threshold and 
                close_val > ema_200_1w_aligned[i]):
                position = 1
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 or closes below L3
            if close_val >= H3_aligned[i] or close_val < L3_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
    
    return signals

name = "12h_Camarilla_L3_Long_WeeklyTrend"
timeframe = "12h"
leverage = 1.0