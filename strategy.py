#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6h timeframe, use Camarilla pivot levels from 1d timeframe for mean reversion and breakout signals.
Buy near S3/S4 support with bullish EMA alignment on 1d, sell near R3/R4 resistance with bearish EMA alignment.
Volume confirmation required (>1.5x average). Targets 12-37 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
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
    
    # EMA for trend filter on 6h
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50 = close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get daily data for Camarilla pivots (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's OHLC
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    # where C = (H+L+CLOSE)/3 of previous day
    
    # We need previous day's OHLC, so we shift by 1
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Calculate Camarilla levels
    r4 = pivot + (range_hl * 1.1 / 2.0)
    r3 = pivot + (range_hl * 1.1 / 4.0)
    s3 = pivot - (range_hl * 1.1 / 4.0)
    s4 = pivot - (range_hl * 1.1 / 2.0)
    
    # Align to 6h timeframe (shifted by 1 day to avoid look-ahead)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # EMA trend filter on 1d (using previous day's EMA to avoid look-ahead)
    daily_close = df_1d['close'].values
    daily_close_s = pd.Series(daily_close)
    ema20_1d = daily_close_s.ewm(span=20, min_periods=20, adjust=False).mean().values
    ema50_1d = daily_close_s.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align EMAs to 6h timeframe
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA50 warmup
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(ema50[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0 or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema20_1d_aligned[i]) or np.isnan(ema50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter from 1d: up if EMA20 > EMA50, down if EMA20 < EMA50
        trend_up = ema20_1d_aligned[i] > ema50_1d_aligned[i]
        trend_down = ema20_1d_aligned[i] < ema50_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit on trend reversal (EMA20 < EMA50)
            if ema20[i] < ema50[i]:
                exit_long = True
            # Exit when price reaches R3 (take profit)
            elif close[i] >= r3_aligned[i]:
                exit_long = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit on trend reversal (EMA20 > EMA50)
            if ema20[i] > ema50[i]:
                exit_short = True
            # Exit when price reaches S3 (take profit)
            elif close[i] <= s3_aligned[i]:
                exit_short = True
            # Exit when volume drops below average
            elif volume[i] < vol_ma[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price near S3/S4 support, bullish trend, volume confirmation
            near_support = (close[i] <= s3_aligned[i] * 1.02) and (close[i] >= s4_aligned[i] * 0.98)
            long_entry = near_support and trend_up and vol_confirm
            
            # Short entry: price near R3/R4 resistance, bearish trend, volume confirmation
            near_resistance = (close[i] >= r3_aligned[i] * 0.98) and (close[i] <= r4_aligned[i] * 1.02)
            short_entry = near_resistance and trend_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals