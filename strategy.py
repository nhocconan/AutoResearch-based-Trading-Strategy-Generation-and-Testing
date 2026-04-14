#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla Pivot Breakout with Volume Confirmation and Weekly Trend Filter
# Uses Camarilla pivot levels from daily data for precise entry/exit points
# Weekly EMA (50) provides trend filter to avoid counter-trend trades
# Volume confirmation (volume > 1.5x average) filters false breakouts
# Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels for each day
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # where C = (H+L+C)/3 (typical price), H = high, L = low, C = close of previous day
    phigh = df_1d['high'].values
    plow = df_1d['low'].values
    pclose = df_1d['close'].values
    
    # Pivot point (typical price)
    pivot = (phigh + plow + pclose) / 3.0
    # Range
    rng = phigh - plow
    
    # Camarilla levels
    r4 = pivot + rng * 1.1 / 2.0
    r3 = pivot + rng * 1.1 / 4.0
    r2 = pivot + rng * 1.1 / 6.0
    r1 = pivot + rng * 1.1 / 12.0
    s1 = pivot - rng * 1.1 / 12.0
    s2 = pivot - rng * 1.1 / 6.0
    s3 = pivot - rng * 1.1 / 4.0
    s4 = pivot - rng * 1.1 / 2.0
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate average volume (20-period) for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 20  # for volume MA
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1[i//12]) or np.isnan(s1[i//12]) or  # daily indices
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        daily_idx = i // 12  # 12 bars per day (12h timeframe)
        
        # Get daily Camarilla levels for previous day (to avoid look-ahead)
        if daily_idx >= 1:
            prev_daily_idx = daily_idx - 1
            r1_val = r1[prev_daily_idx]
            r2_val = r2[prev_daily_idx]
            r3_val = r3[prev_daily_idx]
            r4_val = r4[prev_daily_idx]
            s1_val = s1[prev_daily_idx]
            s2_val = s2[prev_daily_idx]
            s3_val = s3[prev_daily_idx]
            s4_val = s4[prev_daily_idx]
        else:
            # Not enough history, skip
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA
        above_week_ema = price > ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 with volume confirmation and uptrend
            if (price > r3_val and 
                volume[i] > vol_ma[i] * 1.5 and 
                above_week_ema):
                position = 1
                signals[i] = position_size
            # Short: price breaks below S3 with volume confirmation and downtrend
            elif (price < s3_val and 
                  volume[i] > vol_ma[i] * 1.5 and 
                  not above_week_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches R4 (profit target) or falls below R1 (stop) or trend changes
            if (price >= r4_val or 
                price <= r1_val or 
                price < ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches S4 (profit target) or rises above S1 (stop) or trend changes
            if (price <= s4_val or 
                price >= s1_val or 
                price > ema_1w_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_WeekTrend"
timeframe = "12h"
leverage = 1.0