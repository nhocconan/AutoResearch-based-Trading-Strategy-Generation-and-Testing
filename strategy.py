#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy combining 1w Pivot Point support/resistance with volume confirmation
# - Uses weekly pivot levels (S1, S2, R1, R2) calculated from prior week's OHLC
# - Long entries when price bounces off S1/S2 with volume confirmation in bullish weekly context
# - Short entries when price rejects R1/R2 with volume confirmation in bearish weekly context
# - Weekly trend filter: only take longs above weekly EMA20, shorts below weekly EMA20
# - Designed to work in ranging markets (bounces at S1/S2) and trending markets (breaks at R2/S2)
# - Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "6h_WeeklyPivot_S1S2_R1R2_VolumeTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior week's OHLC
    # P = (H + L + C) / 3
    # S1 = 2*P - H, S2 = P - (H - L)
    # R1 = 2*P - L, R2 = P + (H - L)
    prev_weekly_high = df_1w['high'].shift(1)
    prev_weekly_low = df_1w['low'].shift(1)
    prev_weekly_close = df_1w['close'].shift(1)
    
    pivot = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
    S1 = 2 * pivot - prev_weekly_high
    S2 = pivot - (prev_weekly_high - prev_weekly_low)
    R1 = 2 * pivot - prev_weekly_low
    R2 = pivot + (prev_weekly_high - prev_weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    S1_6h = align_htf_to_ltf(prices, df_1w, S1.values)
    S2_6h = align_htf_to_ltf(prices, df_1w, S2.values)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1.values)
    R2_6h = align_htf_to_ltf(prices, df_1w, R2.values)
    
    # Weekly EMA20 for trend filter
    weekly_ema_20 = pd.Series(df_1w['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema_20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_20)
    
    # Volume filters
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)  # Volume confirmation threshold
    
    # Session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after warmup
        # Skip if any critical value is NaN or outside session
        if (np.isnan(S1_6h[i]) or np.isnan(S2_6h[i]) or np.isnan(R1_6h[i]) or np.isnan(R2_6h[i]) or
            np.isnan(weekly_ema_20_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bounce off S1/S2 with volume in bullish weekly context
            if close[i] >= S1_6h[i] * 0.999 and close[i] <= S1_6h[i] * 1.001 and volume_filter[i]:
                if close[i] > weekly_ema_20_aligned[i]:  # Above weekly EMA20
                    signals[i] = 0.25
                    position = 1
            elif close[i] >= S2_6h[i] * 0.999 and close[i] <= S2_6h[i] * 1.001 and volume_filter[i]:
                if close[i] > weekly_ema_20_aligned[i]:  # Above weekly EMA20
                    signals[i] = 0.25
                    position = 1
            # Short setup: rejection at R1/R2 with volume in bearish weekly context
            elif close[i] <= R1_6h[i] * 1.001 and close[i] >= R1_6h[i] * 0.999 and volume_filter[i]:
                if close[i] < weekly_ema_20_aligned[i]:  # Below weekly EMA20
                    signals[i] = -0.25
                    position = -1
            elif close[i] <= R2_6h[i] * 1.001 and close[i] >= R2_6h[i] * 0.999 and volume_filter[i]:
                if close[i] < weekly_ema_20_aligned[i]:  # Below weekly EMA20
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price reaches R1 (profit target) or breaks below S2 (stop)
            if close[i] >= R1_6h[i] * 0.999:  # Take profit at R1
                signals[i] = 0.0
                position = 0
            elif close[i] < S2_6h[i]:  # Stop loss if breaks below S2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price reaches S1 (profit target) or breaks above R2 (stop)
            if close[i] <= S1_6h[i] * 1.001:  # Take profit at S1
                signals[i] = 0.0
                position = 0
            elif close[i] > R2_6h[i]:  # Stop loss if breaks above R2
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals