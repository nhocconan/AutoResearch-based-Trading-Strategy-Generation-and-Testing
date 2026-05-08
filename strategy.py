# 6h_WeeklyPivot_R3S4_Breakout_1dTrend_Volume
# Hypothesis: Use weekly pivot R3/S4 as breakout levels with 1d EMA trend filter and volume confirmation.
# In bull markets, price breaks above R3 with strength; in bear markets, breaks below S4 with momentum.
# Weekly pivot provides institutional reference points; 1d EMA filters counter-trend noise; volume avoids false breakouts.
# Targets 15-30 trades/year on 6b timeframe to minimize fee drag while capturing significant moves.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_WeeklyPivot_R3S4_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get weekly data for pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate weekly pivot points and levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot = (H + L + C) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Support 1 = (2 * Pivot) - High
    s1_1w = (2 * pivot_1w) - high_1w
    # Resistance 1 = (2 * Pivot) - Low
    r1_1w = (2 * pivot_1w) - low_1w
    # Support 2 = Pivot - (High - Low)
    s2_1w = pivot_1w - (high_1w - low_1w)
    # Resistance 2 = Pivot + (High - Low)
    r2_1w = pivot_1w + (high_1w - low_1w)
    # Support 3 = Low - 2*(High - Pivot)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    # Resistance 3 = High + 2*(Pivot - Low)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    # Support 4 = S3 - (R2 - S2)
    s4_1w = s3_1w - (r2_1w - s2_1w)
    # Resistance 4 = R3 + (R2 - S2)
    r4_1w = r3_1w + (r2_1w - s2_1w)
    
    # Align weekly levels to 6h timeframe
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Volume confirmation - 24-period average volume (4 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above R3 with trend and volume confirmation
            if (close[i] > r3_1w_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and
                vol_ratio[i] > 1.8):
                # Avoid chasing too far beyond R4
                if close[i] <= r4_1w_aligned[i] * 1.02:
                    signals[i] = 0.25
                    position = 1
            # Short: break below S3 with trend and volume confirmation
            elif (close[i] < s3_1w_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and
                  vol_ratio[i] > 1.8):
                # Avoid chasing too far beyond S4
                if close[i] >= s4_1w_aligned[i] * 0.98:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price below pivot OR below EMA34
            if close[i] < pivot_1w_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price above pivot OR above EMA34
            if close[i] > pivot_1w_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals