#!/usr/bin/env python3
"""
Hypothesis: 1d Camarilla R1/S1 breakout with volume confirmation and 1w EMA20 trend filter.
Long when price breaks above Camarilla R1 AND volume > 2.0x 20-period average AND price > 1w EMA20 (bullish trend).
Short when price breaks below Camarilla S1 AND volume > 2.0x 20-period average AND price < 1w EMA20 (bearish trend).
Exit when price crosses the 1w EMA20 in opposite direction.
Camarilla levels provide precise intraday support/resistance, volume confirmation reduces false breakouts,
and 1w EMA20 filters for longer-term trend. Designed for low trade frequency (7-25/year) on 1d timeframe
to minimize fee drag while capturing strong breakouts in both bull and bear markets.
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
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA20 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA20 on 1w timeframe
    close_1w_series = pd.Series(close_1w)
    ema_20_1w = close_1w_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate Camarilla levels on 1d timeframe
    # Camarilla R1 = close + (high - low) * 1.1 / 12
    # Camarilla S1 = close - (high - low) * 1.1 / 12
    high_low_range = high_1d - low_1d
    camarilla_r1 = close_1d + high_low_range * 1.1 / 12
    camarilla_s1 = close_1d - high_low_range * 1.1 / 12
    
    # Calculate volume average (20-period) on 1d
    volume_1d_series = pd.Series(volume_1d)
    volume_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (prices)
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_20 = ema_20_1w_aligned[i]
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 2.0x avg AND price > 1w EMA20 (bullish trend)
            if high_price > r1 and vol > 2.0 * vol_ma and price > ema_20:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 2.0x avg AND price < 1w EMA20 (bearish trend)
            elif low_price < s1 and vol > 2.0 * vol_ma and price < ema_20:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1w EMA20
            if price < ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1w EMA20
            if price > ema_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Volume_1wEMA20_Filter"
timeframe = "1d"
leverage = 1.0