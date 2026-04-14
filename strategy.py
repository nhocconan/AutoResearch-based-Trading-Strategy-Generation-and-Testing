#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Camarilla Pivot reversal with weekly trend filter and volume confirmation
# Long when price touches L3 level AND weekly close > weekly EMA50 AND volume > 2x 24-period average
# Short when price touches H3 level AND weekly close < weekly EMA50 AND volume > 2x 24-period average
# Exit when price reaches opposite H3/L3 level or reverses from touch
# Uses Camarilla levels for institutional reversal zones, weekly trend for direction filter, volume for confirmation
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag while capturing reversals

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Camarilla levels from previous day (requires daily data)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels: based on previous day's range
    # H3 = C + (H-L)*1.1/4, L3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    camarilla_h3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    camarilla_l3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate volume average for confirmation (24-period = 2 days of 12h data)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_threshold = vol_avg[i] * 2.0
        
        if position == 0:
            # Long setup: price touches L3 AND weekly uptrend AND volume confirmation
            if (price <= camarilla_l3_aligned[i] * 1.001 and  # Allow small buffer for touch
                price > camarilla_l3_aligned[i] * 0.999 and
                close_1w[-1] > ema50_1w[-1] if len(close_1w) > 0 else False and  # Weekly trend (use last known)
                vol > vol_threshold):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H3 AND weekly downtrend AND volume confirmation
            elif (price >= camarilla_h3_aligned[i] * 0.999 and  # Allow small buffer for touch
                  price <= camarilla_h3_aligned[i] * 1.001 and
                  close_1w[-1] < ema50_1w[-1] if len(close_1w) > 0 else False and  # Weekly trend
                  vol > vol_threshold):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H3 level or shows rejection from L3 area
            if price >= camarilla_h3_aligned[i] * 0.999:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L3 level or shows rejection from H3 area
            if price <= camarilla_l3_aligned[i] * 1.001:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_L3H3_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0