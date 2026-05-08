#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_Volume_v2"
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
    
    # Get daily data for Camarilla pivot, trend, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla pivot calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels R1, R2, S1, S2
    R1 = pivot + (range_hl * 1.0 / 12)
    R2 = pivot + (range_hl * 1.1 / 12)
    S1 = pivot - (range_hl * 1.0 / 12)
    S2 = pivot - (range_hl * 1.1 / 12)
    
    # Trend filter: 34-period EMA on daily close
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike: current 4h volume vs 24-period average (4 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Align all indicators to 4h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    R2_aligned = align_htf_to_ltf(prices, df_1d, R2)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    S2_aligned = align_htf_to_ltf(prices, df_1d, S2)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(R1_aligned[i]) or np.isnan(R2_aligned[i]) or 
            np.isnan(S1_aligned[i]) or np.isnan(S2_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = vol_ratio[i]
        ema = ema_34_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 with volume + above daily EMA34
            if (price > R1_aligned[i] and 
                vol > 1.8 and 
                price > ema):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 with volume + below daily EMA34
            elif (price < S1_aligned[i] and 
                  vol > 1.8 and 
                  price < ema):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S1 or volume drops
            if (price < S1_aligned[i] or vol < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price breaks above R1 or volume drops
            if (price > R1_aligned[i] or vol < 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals