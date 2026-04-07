#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: On 6-hour timeframe, use Camarilla pivot levels derived from 1-day high/low/close to identify reversal and breakout zones. Fade at R3/S3 (mean reversion) when price is outside daily EMA200 trend, breakout continuation at R4/S4 (trend follow) when aligned with daily EMA200 trend. Volume confirmation required. Designed for low frequency (12-37 trades/year) to avoid fee drag while capturing both mean reversion in ranges and trend continuation in trends. Works in bull (buy R4 breakouts in uptrend, sell S3 reversals in overbought) and bear (sell S4 breakdowns in downtrend, buy R3 reversals in oversold) by using daily trend filter.
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
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Calculate daily EMA200 for trend filter
    d_ema200 = pd.Series(d_close).ewm(span=200, adjust=False).mean().values
    d_ema200_aligned = align_htf_to_ltf(prices, df_1d, d_ema200)
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = close + (high - low) * 1.1/2
    # R3 = close + (high - low) * 1.1/4
    # R2 = close + (high - low) * 1.1/6
    # R1 = close + (high - low) * 1.1/12
    # PP = (high + low + close) / 3
    # S1 = close - (high - low) * 1.1/12
    # S2 = close - (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/4
    # S4 = close - (high - low) * 1.1/2
    
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_d_high = np.concatenate([[np.nan], d_high[:-1]])
    prev_d_low = np.concatenate([[np.nan], d_low[:-1]])
    prev_d_close = np.concatenate([[np.nan], d_close[:-1]])
    
    # Calculate pivots
    camarilla_pp = (prev_d_high + prev_d_low + prev_d_close) / 3.0
    camarilla_range = prev_d_high - prev_d_low
    
    r4 = prev_d_close + camarilla_range * 1.1 / 2.0
    r3 = prev_d_close + camarilla_range * 1.1 / 4.0
    s3 = prev_d_close - camarilla_range * 1.1 / 4.0
    s4 = prev_d_close - camarilla_range * 1.1 / 2.0
    
    # Align to 6h timeframe
    camarilla_pp_aligned = align_htf_to_ltf(prices, df_1d, camarilla_pp)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after EMA200 warmup
        # Skip if daily data not available
        if np.isnan(d_ema200_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Determine trend based on price vs daily EMA200
        uptrend = close[i] > d_ema200_aligned[i]
        downtrend = close[i] < d_ema200_aligned[i]
        
        # Volume confirmation: current volume > 1.3x 20-period average
        vol_confirm = volume[i] > 1.3 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price reaches S3 (mean reversion target) or R4 (breakout exhaustion)
            if close[i] <= s3_aligned[i] or close[i] >= r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price reaches R3 (mean reversion target) or S4 (breakout exhaustion)
            if close[i] >= r3_aligned[i] or close[i] <= s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Mean reversion fade at R3/S3 when counter-trend
            fade_short = (close[i] >= r3_aligned[i]) and downtrend and vol_confirm
            fade_long = (close[i] <= s3_aligned[i]) and uptrend and vol_confirm
            
            # Breakout continuation at R4/S4 when with-trend
            breakout_long = (close[i] >= r4_aligned[i]) and uptrend and vol_confirm
            breakout_short = (close[i] <= s4_aligned[i]) and downtrend and vol_confirm
            
            if fade_long or breakout_long:
                position = 1
                signals[i] = 0.25
            elif fade_short or breakout_short:
                position = -1
                signals[i] = -0.25
    
    return signals