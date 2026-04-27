#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Daily Camarilla levels (based on previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Resistance and Support levels (R1, S1)
    R1 = pivot + (range_hl * 1.1 / 6)  # R1 level
    S1 = pivot - (range_hl * 1.1 / 6)  # S1 level
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend filter
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # 1h volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.20   # Position size: 20% of capital
    
    # Warmup: need daily pivots, 4h EMA, and volume data
    start_idx = max(2, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or 
            np.isnan(ema20_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Apply session filter: only trade 08-20 UTC
        if hours[i] < 8 or hours[i] > 20:
            signals[i] = 0.0
            continue
        
        ema_trend = ema20_4h_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 2.0 * 20-period average (volume spike)
        vol_filter = vol_current > (vol_ma_val * 2.0)
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and volume spike
            if close[i] > R1_aligned[i] and close[i] > ema_trend and vol_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with 4h downtrend and volume spike
            elif close[i] < S1_aligned[i] and close[i] < ema_trend and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below S1 or 4h trend turns down
            if close[i] < S1_aligned[i] or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above R1 or 4h trend turns up
            if close[i] > R1_aligned[i] or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume_Session"
timeframe = "1h"
leverage = 1.0