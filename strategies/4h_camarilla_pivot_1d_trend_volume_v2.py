#!/usr/bin/env python3
# 4h_camarilla_pivot_1d_trend_volume_v2
# Hypothesis: On 4h timeframe, use Camarilla pivot levels from 1d with tighter entry conditions to reduce trade frequency.
# Long when price closes above Camarilla resistance level R4 with volume > 1.8x average and 1d uptrend.
# Short when price closes below Camarilla support level S4 with volume > 1.8x average and 1d downtrend.
# Exit when price returns to Camarilla pivot point.
# Uses R4/S4 levels (more extreme than R3/S3) for fewer, higher-quality trades.
# Target: 15-30 trades/year to avoid fee drag while maintaining edge in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (based on previous day's range)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 2:
        return np.zeros(n)
    
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate pivot and support/resistance levels
    pivot = (daily_high + daily_low + daily_close) / 3
    range_hl = daily_high - daily_low
    r4 = pivot + (range_hl * 1.5)  # More extreme than R3
    s4 = pivot - (range_hl * 1.5)  # More extreme than S3
    
    # Align to 4h timeframe
    pivot_4h = align_htf_to_ltf(prices, df_daily, pivot)
    r4_4h = align_htf_to_ltf(prices, df_daily, r4)
    s4_4h = align_htf_to_ltf(prices, df_daily, s4)
    
    # Calculate 1d trend filter: EMA20
    daily_ema20 = pd.Series(daily_close).ewm(span=20, min_periods=20, adjust=False).mean().values
    daily_ema20_4h = align_htf_to_ltf(prices, df_daily, daily_ema20)
    
    # Volume confirmation: 20-period average on 4h
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_4h[i]) or np.isnan(r4_4h[i]) or np.isnan(s4_4h[i]) or \
           np.isnan(daily_ema20_4h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to pivot point
            if close[i] <= pivot_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot point
            if close[i] >= pivot_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.8x average volume (stricter)
            volume_ok = volume[i] > 1.8 * avg_volume[i]
            
            # Daily trend filter
            daily_uptrend = close[i] > daily_ema20_4h[i]
            daily_downtrend = close[i] < daily_ema20_4h[i]
            
            # Long entry: price closes above R4 with volume and uptrend
            if close[i] > r4_4h[i] and volume_ok and daily_uptrend:
                position = 1
                signals[i] = 0.25
            # Short entry: price closes below S4 with volume and downtrend
            elif close[i] < s4_4h[i] and volume_ok and daily_downtrend:
                position = -1
                signals[i] = -0.25
    
    return signals