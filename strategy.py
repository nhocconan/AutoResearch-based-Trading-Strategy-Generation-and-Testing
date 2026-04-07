#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Weekly Pivot + Volume + ATR Trend Filter
# Hypothesis: Trade reversals at weekly pivot levels (R1/S1) with volume confirmation
# and trend filter using daily ATR-based trend. Works in bull/bear by fading extremes
# in ranging markets and following trend in trending markets. Target: 15-30 trades/year.

name = "1d_weekly_pivot_volume_atr_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot points
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week)
    close_weekly = df_weekly['close'].values
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    
    # Prior week's data for pivot calculation
    prev_close = np.roll(close_weekly, 1)
    prev_high = np.roll(high_weekly, 1)
    prev_low = np.roll(low_weekly, 1)
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    
    # Standard pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align weekly pivots to daily
    pivot_daily = align_htf_to_ltf(prices, df_weekly, pivot)
    r1_daily = align_htf_to_ltf(prices, df_weekly, r1)
    s1_daily = align_htf_to_ltf(prices, df_weekly, s1)
    r2_daily = align_htf_to_ltf(prices, df_weekly, r2)
    s2_daily = align_htf_to_ltf(prices, df_weekly, s2)
    
    # Daily ATR-based trend filter (using 14-period ATR)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Trend: price vs ATR-smoothed median
    med_price = (high + low + close) / 3.0
    med_smoothed = pd.Series(med_price).rolling(window=20, min_periods=20).median().values
    trend_up = med_price > med_smoothed + 0.5 * atr
    trend_down = med_price < med_smoothed - 0.5 * atr
    
    # Volume filter: volume > 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(pivot_daily[i]) or np.isnan(r1_daily[i]) or np.isnan(s1_daily[i]) or
            np.isnan(r2_daily[i]) or np.isnan(s2_daily[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches S1 or trend turns down
            if low[i] <= s1_daily[i] or trend_down[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches R1 or trend turns up
            if high[i] >= r1_daily[i] or trend_up[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at extremes in ranging markets, follow trend in trending markets
            if vol_ok:
                # In ranging market: fade at R1/S1
                if not (trend_up[i] or trend_down[i]):  # Ranging
                    if low[i] <= s1_daily[i] and close[i] > s1_daily[i]:  # Bounce off S1
                        position = 1
                        signals[i] = 0.25
                    elif high[i] >= r1_daily[i] and close[i] < r1_daily[i]:  # Rejection at R1
                        position = -1
                        signals[i] = -0.25
                # In trending market: pullback to pivot
                else:
                    if trend_up[i] and low[i] <= pivot_daily[i] and close[i] > pivot_daily[i]:
                        position = 1
                        signals[i] = 0.25
                    elif trend_down[i] and high[i] >= pivot_daily[i] and close[i] < pivot_daily[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals