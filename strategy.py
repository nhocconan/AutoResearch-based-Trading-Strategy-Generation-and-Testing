#!/usr/bin/env python3
"""
12h_12hr_Camarilla_R1_S1_Breakout_1dTrend_Volume
Hypothesis: 12-hour breakouts of Camarilla R1/S1 levels with daily trend filter and volume confirmation.
Camarilla levels provide precise support/resistance from prior day's range. Works in bull/bear via
trend filter: only trade long in uptrend (price > EMA50) and short in downtrend (price < EMA50).
Volume filter avoids low-liquidity false breakouts. Target: 15-25 trades/year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day Camarilla levels (based on prior day OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 1d bar: R1, S1
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_R1 = df_1d['close'] + 1.1 * (df_1d['high'] - df_1d['low']) / 12
    camarilla_S1 = df_1d['close'] - 1.1 * (df_1d['high'] - df_1d['low']) / 12
    
    # Align to 12h timeframe (wait for prior day's close)
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1.values)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1.values)
    
    # 1-day EMA trend filter (EMA50)
    ema_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        vol_ok = volume_filter[i]
        ema_trend = ema_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume in uptrend
            if price > r1 and vol_ok and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume in downtrend
            elif price < s1 and vol_ok and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Maintain long until price breaks below S1 or trend reverses
            if price < s1 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Maintain short until price breaks above R1 or trend reverses
            if price > r1 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_12hr_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0