#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot with weekly trend filter and volume spike
# Camarilla R3/S3 levels from daily provide mean reversion in range markets
# Weekly trend filter (price vs weekly EMA20) adapts to bull/bear regimes
# Volume spike confirms institutional participation
# Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag
# Works in bull (buy R3 bounces in uptrend) and bear (sell S3 rallies in downtrend)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R3 = C + (H-L) * 1.1/2
    # S3 = C - (H-L) * 1.1/2
    # R4 = C + (H-L) * 1.1
    # S4 = C - (H-L) * 1.1
    
    # Previous day's OHLC
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Handle first day
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    rang = prev_high - prev_low
    
    r3 = pivot + rang * 1.1 / 2.0
    s3 = pivot - rang * 1.1 / 2.0
    r4 = pivot + rang * 1.1
    s4 = pivot - rang * 1.1
    
    # Align Camarilla levels to 6h
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Weekly trend filter: EMA20 on weekly
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or 
            np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or
            np.isnan(ema_20_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not session_mask[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below weekly EMA20
        price_above_weekly_ema = close[i] > ema_20_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Long setup: price at S3/S4 with bullish weekly trend
        near_s3 = abs(close[i] - s3_6h[i]) < (r3_6h[i] - s3_6h[i]) * 0.02  # Within 2% of S3
        near_s4 = abs(close[i] - s4_6h[i]) < (r4_6h[i] - s4_6h[i]) * 0.02  # Within 2% of S4
        at_support = near_s3 or near_s4
        
        long_setup = at_support and price_above_weekly_ema and vol_confirm
        
        # Short setup: price at R3/R4 with bearish weekly trend
        near_r3 = abs(close[i] - r3_6h[i]) < (r3_6h[i] - s3_6h[i]) * 0.02  # Within 2% of R3
        near_r4 = abs(close[i] - r4_6h[i]) < (r4_6h[i] - s4_6h[i]) * 0.02  # Within 2% of R4
        at_resistance = near_r3 or near_r4
        
        short_setup = at_resistance and price_below_weekly_ema and vol_confirm
        
        # Entry logic
        if long_setup and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_setup and position >= 0:
            signals[i] = -0.25
            position = -1
        # Exit logic: opposite setup or extreme levels
        elif position == 1 and (short_setup or close[i] > r4_6h[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (long_setup or close[i] < s4_6h[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_Camarilla_R3S3_WeeklyTrend_Volume"
timeframe = "6h"
leverage = 1.0