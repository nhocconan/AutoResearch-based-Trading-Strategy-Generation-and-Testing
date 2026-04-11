#!/usr/bin/env python3
"""
1h_4h_1d_camarilla_breakout_volume_v1
Strategy: 1h Camarilla pivot breakout with volume confirmation and 4h/1d trend filter
Timeframe: 1h
Leverage: 1.0
Hypothesis: Uses 1h Camarilla pivot levels (H3/L3) for breakout entries, confirmed by volume spike (>1.5x average) and filtered by 4h EMA50 and 1d EMA200 trend alignment. Designed to capture momentum breaks in trending markets while avoiding false signals in chop. Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered out). Target: 60-150 total trades over 4 years (15-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_camarilla_breakout_volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # 1h typical price for Camarilla calculation
    typical_price = (high + low + close) / 3.0
    
    # Previous day's OHLC for Camarilla (using 1h data - approximate daily from last 24h)
    # We'll use rolling 24-period high/low/close for daily approximation
    if n >= 24:
        prev_high = np.roll(high, 24)
        prev_low = np.roll(low, 24)
        prev_close = np.roll(close, 24)
        # For first 24 bars, use available data
        prev_high[:24] = high[0]
        prev_low[:24] = low[0]
        prev_close[:24] = close[0]
    else:
        prev_high = high
        prev_low = low
        prev_close = close
    
    # Camarilla levels: H3/L3 = close +- 1.1*(high-low)/2
    rang = prev_high - prev_low
    h3 = prev_close + 1.1 * rang / 2.0
    l3 = prev_close - 1.1 * rang / 2.0
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    # 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):  # Start after we have enough data for Camarilla
        # Skip if any required data is invalid
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        price_close = close[i]
        
        # Trend filters: 4h EMA50 and 1d EMA200 alignment
        uptrend_4h = price_close > ema_50_4h_aligned[i]
        uptrend_1d = price_close > ema_200_1d_aligned[i]
        downtrend_4h = price_close < ema_50_4h_aligned[i]
        downtrend_1d = price_close < ema_200_1d_aligned[i]
        
        # Breakout conditions
        breakout_up = price_close > h3[i]
        breakout_down = price_close < l3[i]
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: upward breakout with volume in uptrend (both timeframes agree)
        long_signal = breakout_up and vol_confirmed and uptrend_4h and uptrend_1d
        
        # Short: downward breakout with volume in downtrend (both timeframes agree)
        short_signal = breakout_down and vol_confirmed and downtrend_4h and downtrend_1d
        
        # Exit when price returns to previous day's close (pivot point)
        # Approximate pivot as (prev_high + prev_low + prev_close)/3
        pivot_point = (prev_high[i] + prev_low[i] + prev_close[i]) / 3.0
        exit_long = position == 1 and price_close < pivot_point
        exit_short = position == -1 and price_close > pivot_point
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals