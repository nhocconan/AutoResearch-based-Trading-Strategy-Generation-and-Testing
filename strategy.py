# 6h_Weekly_Pivot_Swing_Rejection
# Hypothesis: On 6h timeframe, price often rejects at weekly pivot levels (R1/S1, R2/S2) during
# weekly trend exhaustion. Enter on rejection with confirmation from 1d EMA trend filter and
# volume spike. Works in bull/bear by only taking trades in direction of higher timeframe trend.
# Target: 15-25 trades/year per symbol (~60-100 total over 4 years).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data for pivot calculation
    df_1w = get_htf_ata(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly range
    weekly_range = high_1w - low_1w
    
    # Calculate weekly pivot points from previous week
    prev_close = np.roll(close_1w, 1)
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_range = np.roll(weekly_range, 1)
    
    # Set first week values to NaN
    prev_close[0] = np.nan
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_range[0] = np.nan
    
    # Calculate weekly R1, S1, R2, S2 (standard pivot formulas)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = (2 * pivot) - prev_low
    s1 = (2 * pivot) - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Load daily data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50 = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume spike filter (24-period on 6h ≈ 6 days)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > 1.5 * vol_ma24
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Align weekly pivot levels to 6h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1w, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1w, s2)
    
    # Align daily EMA to 6h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(100, n):
        # Skip if data not ready or outside session
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma24[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: Reject at S1/S2 with volume spike and uptrend
            long_reject_s1 = (low[i] <= s1_aligned[i] and close[i] > s1_aligned[i])
            long_reject_s2 = (low[i] <= s2_aligned[i] and close[i] > s2_aligned[i])
            uptrend = close[i] > ema_50_aligned[i]
            
            if (long_reject_s1 or long_reject_s2) and vol_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short setup: Reject at R1/R2 with volume spike and downtrend
            elif (high[i] >= r1_aligned[i] and close[i] < r1_aligned[i]) or \
                 (high[i] >= r2_aligned[i] and close[i] < r2_aligned[i]):
                if vol_spike[i] and close[i] < ema_50_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        else:
            # Exit: Price moves back through the pivot level
            if position == 1:
                if close[i] < pivot[i]:  # Use current week's pivot for exit
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Weekly_Pivot_Swing_Rejection"
timeframe = "6h"
leverage = 1.0