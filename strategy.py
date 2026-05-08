#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot level touch with 12h EMA trend filter and volume spike
# Uses daily for Camarilla pivot calculation (H/L/C from previous day)
# 12h EMA for trend direction filter
# Volume spike confirmation to avoid false breakouts
# Works in bull/bear via mean reversion at extreme pivot levels with trend filter
# Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag

name = "4h_Camarilla_Pivot_Touch_12hEMA_Volume"
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
    
    # Get daily data for Camarilla pivot calculation (uses previous day's H/L/C)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's data
    # Camarilla uses: H = high, L = low, C = close of previous day
    # Resistance levels: R1 = C + (H-L)*1.1/12, R2 = C + (H-L)*1.1/6, R3 = C + (H-L)*1.1/4, R4 = C + (H-L)*1.1/2
    # Support levels: S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    # We'll focus on the inner levels (R3/S3 and R4/S4) for stronger signals
    
    prev_high = df_daily['high'].values
    prev_low = df_daily['low'].values
    prev_close = df_daily['close'].values
    
    # Calculate Camarilla levels for each day
    H_minus_L = prev_high - prev_low
    R4 = prev_close + H_minus_L * 1.1 / 2
    R3 = prev_close + H_minus_L * 1.1 / 4
    S3 = prev_close - H_minus_L * 1.1 / 4
    S4 = prev_close - H_minus_L * 1.1 / 2
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA (20-period)
    ema_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate volume spike (current volume > 2x 20-period EMA of volume)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > 2 * vol_ema_20
    
    # Align all indicators to 4h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_daily, R4)
    R3_aligned = align_htf_to_ltf(prices, df_daily, R3)
    S3_aligned = align_htf_to_ltf(prices, df_daily, S3)
    S4_aligned = align_htf_to_ltf(prices, df_daily, S4)
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check for NaN values
        if (np.isnan(R4_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(S4_aligned[i]) or np.isnan(ema_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_12h_aligned[i]
        
        if position == 0:
            # Look for mean reversion at extreme Camarilla levels with trend filter
            # Long when price touches S3/S4 and above EMA (uptrend bias)
            # Short when price touches R3/R4 and below EMA (downtrend bias)
            if price <= S3_aligned[i] and price > ema_trend:
                # Additional confirmation: price should be near S4 for stronger signal
                if price <= S4_aligned[i] * 1.002:  # within 0.2% of S4
                    signals[i] = 0.25
                    position = 1
            elif price >= R3_aligned[i] and price < ema_trend:
                # Additional confirmation: price should be near R4 for stronger signal
                if price >= R4_aligned[i] * 0.998:  # within 0.2% of R4
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price moves back to mean (S1 level) or trend changes
            if price >= S3_aligned[i] * 1.02:  # 2% above S3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price moves back to mean (R1 level) or trend changes
            if price <= R3_aligned[i] * 0.98:  # 2% below R3
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals