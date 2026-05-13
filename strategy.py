#!/usr/bin/env python3
name = "4h_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1D data ONCE for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R1, S1) from previous day
    # Using standard formula: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shifted by 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # First day has no previous day, set to NaN
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for previous day
    camarilla_R1 = prev_close + 1.1 * (prev_high - prev_low) / 12
    camarilla_S1 = prev_close - 1.1 * (prev_high - prev_low) / 12
    
    # Align 1D Camarilla levels to 4H timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # 1D EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 4H volume spike filter (volume > 1.5 * 20-period average)
    volume_series = pd.Series(prices['volume'].values)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = prices['volume'].values > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA34 warmup
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            signals[i] = 0.0
            continue
        
        price = prices['close'].iloc[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        ema34 = ema34_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        # Trend filter: price above/below EMA34
        price_above_ema = price > ema34
        price_below_ema = price < ema34
        
        if position == 0:
            # LONG: Price breaks above R1 + uptrend + volume spike
            if price > r1 and price_above_ema and vol_spike:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S1 + downtrend + volume spike
            elif price < s1 and price_below_ema and vol_spike:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below S1 or trend changes
            if price < s1 or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 or trend changes
            if price > r1 or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals