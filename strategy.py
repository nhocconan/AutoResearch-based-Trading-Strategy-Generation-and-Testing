# 1d_Weekly_Camarilla_Pullback_Reversal
# Hypothesis: Camarilla pivot levels from weekly timeframe act as strong support/resistance.
# Price tends to pull back to these levels before continuing the trend.
# In bull markets, buy near S3/S4; in bear markets, sell near R3/R4.
# Uses weekly Camarilla levels for structure and daily price action for entry.
# Designed for low trade frequency (10-25/year) to minimize fee drag and work in both bull/bear markets.

name = "1d_Weekly_Camarilla_Pullback_Reversal"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given period."""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close
    c = close
    h = high
    l = low
    r4 = c + ((h - l) * 1.1 / 2)
    r3 = c + ((h - l) * 1.1 / 4)
    r2 = c + ((h - l) * 1.1 / 6)
    s2 = c - ((h - l) * 1.1 / 6)
    s3 = c - ((h - l) * 1.1 / 4)
    s4 = c - ((h - l) * 1.1 / 2)
    return r4, r3, r2, s2, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    # Pre-calculate all Camarilla levels
    r4_w = np.full_like(weekly_close, np.nan)
    r3_w = np.full_like(weekly_close, np.nan)
    r2_w = np.full_like(weekly_close, np.nan)
    s2_w = np.full_like(weekly_close, np.nan)
    s3_w = np.full_like(weekly_close, np.nan)
    s4_w = np.full_like(weekly_close, np.nan)
    
    for i in range(len(weekly_close)):
        r4, r3, r2, s2, s3, s4 = calculate_camarilla(weekly_high[i], weekly_low[i], weekly_close[i])
        r4_w[i] = r4
        r3_w[i] = r3
        r2_w[i] = r2
        s2_w[i] = s2
        s3_w[i] = s3
        s4_w[i] = s4
    
    # Align weekly Camarilla levels to daily timeframe
    r4_w_aligned = align_htf_to_ltf(prices, df_1w, r4_w)
    r3_w_aligned = align_htf_to_ltf(prices, df_1w, r3_w)
    r2_w_aligned = align_htf_to_ltf(prices, df_1w, r2_w)
    s2_w_aligned = align_htf_to_ltf(prices, df_1w, s2_w)
    s3_w_aligned = align_htf_to_ltf(prices, df_1w, s3_w)
    s4_w_aligned = align_htf_to_ltf(prices, df_1w, s4_w)
    
    # Daily trend filter: EMA 50
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: 20-day average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for EMA50
    
    for i in range(start_idx, n):
        if np.isnan(r4_w_aligned[i]) or np.isnan(r3_w_aligned[i]) or \
           np.isnan(r2_w_aligned[i]) or np.isnan(s2_w_aligned[i]) or \
           np.isnan(s3_w_aligned[i]) or np.isnan(s4_w_aligned[i]) or \
           np.isnan(ema_50[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long entry: pullback to S3 or S4 in uptrend
            if close[i] > ema_50[i]:  # Uptrend filter
                if (close[i] <= s3_w_aligned[i] * 1.002 and close[i] >= s3_w_aligned[i] * 0.998) or \
                   (close[i] <= s4_w_aligned[i] * 1.002 and close[i] >= s4_w_aligned[i] * 0.998):
                    if vol_confirm:
                        signals[i] = 0.25
                        position = 1
            # Short entry: pullback to R3 or R4 in downtrend
            elif close[i] < ema_50[i]:  # Downtrend filter
                if (close[i] >= r3_w_aligned[i] * 0.998 and close[i] <= r3_w_aligned[i] * 1.002) or \
                   (close[i] >= r4_w_aligned[i] * 0.998 and close[i] <= r4_w_aligned[i] * 1.002):
                    if vol_confirm:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price reaches R2 or closes below EMA50
            if close[i] >= r2_w_aligned[i] * 0.998 or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reaches S2 or closes above EMA50
            if close[i] <= s2_w_aligned[i] * 1.002 or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

#!/usr/bin/env python3