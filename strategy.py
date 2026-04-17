#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND volume > 1.3x 20-period average AND price > 1d EMA50.
Short when price breaks below Camarilla S1 AND volume > 1.3x 20-period average AND price < 1d EMA50.
Exit when price crosses the 1d EMA50 in opposite direction.
Designed for low trade frequency (12-37/year) to minimize fee drag while capturing strong breakouts in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1d data for EMA50 trend filter (HTF)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla pivot levels on 1d timeframe
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Calculate volume average (20-period) on 12h timeframe (primary)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 1.3x avg AND price > 1d EMA50 (bullish trend)
            if high_price > r1 and vol > 1.3 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 1.3x avg AND price < 1d EMA50 (bearish trend)
            elif low_price < s1 and vol > 1.3 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA50
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA50
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1dEMA50_Filter"
timeframe = "12h"
leverage = 1.0