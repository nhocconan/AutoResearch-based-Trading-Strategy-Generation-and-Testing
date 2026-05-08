#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using daily pivot points with volume-weighted price action and trend filter.
# Pivot levels act as institutional support/resistance. Long when price bounces from S1/S2 in uptrend with volume confirmation.
# Short when price rejects R1/R2 in downtrend with volume. Uses 1-week ADX to filter ranging markets.
# Designed for low trade frequency (15-25/year) to avoid whipsaw in chop while capturing institutional reaction at key levels.

name = "6h_PivotReversal_VolumeTrendFilter"
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
    
    # Get daily data for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard floor trader pivots)
    pivot = np.zeros_like(close_1d)
    r1 = np.zeros_like(close_1d)
    s1 = np.zeros_like(close_1d)
    r2 = np.zeros_like(close_1d)
    s2 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        # Prior day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        
        # Pivot point and support/resistance levels
        pivot[i] = (ph + pl + pc) / 3.0
        r1[i] = 2 * pivot[i] - pl
        s1[i] = 2 * pivot[i] - ph
        r2[i] = pivot[i] + (ph - pl)
        s2[i] = pivot[i] - (ph - pl)
    
    # First day has no prior data
    pivot[0] = r1[0] = s1[0] = r2[0] = s2[0] = np.nan
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Get weekly trend filter using ADX to avoid ranging markets
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX(14) on weekly data
    def calculate_adx(high, low, close, period=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] < minus_dm[i]:
                plus_dm[i] = 0
            if minus_dm[i] < plus_dm[i]:
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        atr[period-1] = np.mean(tr[:period])
        plus_dm_sum = np.sum(plus_dm[:period])
        minus_dm_sum = np.sum(minus_dm[:period])
        
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
            plus_dm_sum = plus_dm_sum - (plus_dm_sum/period) + plus_dm[i]
            minus_dm_sum = minus_dm_sum - (minus_dm_sum/period) + minus_dm[i]
            plus_di[i] = 100 * plus_dm_sum / atr[i] if atr[i] != 0 else 0
            minus_di[i] = 100 * minus_dm_sum / atr[i] if atr[i] != 0 else 0
            dx[i] = abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100 if (plus_di[i] + minus_di[i]) != 0 else 0
        
        # Smoothed DX for ADX
        adx = np.zeros_like(high)
        adx[2*period-2] = np.mean(dx[period-1:2*period-1])
        for i in range(2*period-1, len(high)):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
        return adx
    
    adx_1w = calculate_adx(high_1w, low_1w, close_1w, 14)
    # Trending market: ADX > 25
    trending_market = adx_1w > 25
    weekly_trend_filter = align_htf_to_ltf(prices, df_1w, trending_market.astype(float))
    
    # 6x EMA(34) for intermediate trend
    close_series = pd.Series(close)
    ema_34 = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume confirmation: current volume > 2x 20-period EMA
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(34) and indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(ema_34[i]) or
            np.isnan(weekly_trend_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long setup: bounce from S1/S2 in uptrend with volume
            if (weekly_trend_filter[i] > 0.5 and  # Weekly trending market
                ema_34[i] > close[i] and          # Price below EMA (pullback)
                (close[i] <= s1_aligned[i] * 1.005 and close[i] >= s1_aligned[i] * 0.995 or
                 close[i] <= s2_aligned[i] * 1.005 and close[i] >= s2_aligned[i] * 0.995) and  # At S1/S2
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short setup: rejection at R1/R2 in uptrend with volume
            elif (weekly_trend_filter[i] > 0.5 and  # Weekly trending market
                  ema_34[i] < close[i] and          # Price above EMA (pullback)
                  (close[i] >= r1_aligned[i] * 0.995 and close[i] <= r1_aligned[i] * 1.005 or
                   close[i] >= r2_aligned[i] * 0.995 and close[i] <= r2_aligned[i] * 1.005) and  # At R1/R2
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below S2 or lose momentum
            if close[i] < s2_aligned[i] * 0.995 or ema_34[i] < close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above R2 or lose momentum
            if close[i] > r2_aligned[i] * 1.005 or ema_34[i] > close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals