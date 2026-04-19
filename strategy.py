#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Camarilla_R3S3_Fade_Breakout_R4S4_Volume"
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
    
    # Get weekly and daily data once before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly data for directional bias (from previous week)
    weekly_close = df_1w['close'].values
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Daily data for Camarilla levels (from previous day)
    daily_close = df_1d['close'].values
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Weekly trend bias: price above/below weekly midpoint
    weekly_mid = (weekly_high + weekly_low) / 2.0
    weekly_bias = np.where(weekly_close > weekly_mid, 1, -1)
    
    # Daily Camarilla levels from previous day
    prev_daily_close = np.roll(daily_close, 1)
    prev_daily_high = np.roll(daily_high, 1)
    prev_daily_low = np.roll(daily_low, 1)
    prev_daily_close[0] = np.nan
    prev_daily_high[0] = np.nan
    prev_daily_low[0] = np.nan
    
    # Pivot = (H + L + C) / 3
    pivot = (prev_daily_high + prev_daily_low + prev_daily_close) / 3.0
    # R3 = C + (H - L) * 1.1 / 4
    r3 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 4.0
    # S3 = C - (H - L) * 1.1 / 4
    s3 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 4.0
    # R4 = C + (H - L) * 1.1 / 2
    r4 = prev_daily_close + (prev_daily_high - prev_daily_low) * 1.1 / 2.0
    # S4 = C - (H - L) * 1.1 / 2
    s4 = prev_daily_close - (prev_daily_high - prev_daily_low) * 1.1 / 2.0
    
    # Align weekly bias to 6h
    weekly_bias_6h = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Align daily Camarilla levels to 6h
    pivot_6h = align_htf_to_ltf(prices, df_1d, pivot)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Time filter: 08-20 UTC (active hours)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    time_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if not time_filter[i]:
            signals[i] = 0.0
            continue
            
        if np.isnan(weekly_bias_6h[i]) or np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or \
           np.isnan(s3_6h[i]) or np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        bias = weekly_bias_6h[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = vol > 1.8 * vol_ma
        
        if position == 0:
            # Fade at R3/S3 in ranging markets (when bias is weak)
            # Breakout at R4/S4 in trending markets (when bias is strong)
            if bias > 0:  # Weekly bullish bias
                # Fade at R3 (resistance) in weak trends
                if price < r3_6h[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
                # Breakout above R4 in strong trends
                elif price > r4_6h[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
            else:  # Weekly bearish bias
                # Fade at S3 (support) in weak trends
                if price > s3_6h[i] and volume_spike:
                    signals[i] = 0.25
                    position = 1
                # Breakdown below S4 in strong trends
                elif price < s4_6h[i] and volume_spike:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: price returns to S3 (fade level) or breaks below S4 (stop)
            if price < s3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to R3 (fade level) or breaks above R4 (stop)
            if price > r3_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals