#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d trend filter and volume confirmation.
Long when price breaks above R3 with 1d EMA50 uptrend and volume > 1.5x average.
Short when price breaks below S3 with 1d EMA50 downtrend and volume > 1.5x average.
Exit when price returns to the Camarilla H-L range (between H3 and L3) or volume drops.
Uses proven Camarilla structure with 1d trend filter to avoid counter-trend trades.
Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
Works in bull markets via breakout continuation and bear markets via fade at extremes.
"""

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
    
    # Get 6h data for Camarilla calculation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Get 1d data for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for 6h
    # Based on previous 6h bar's OHLC
    camarilla_high = np.roll(high_6h, 1)
    camarilla_low = np.roll(low_6h, 1)
    camarilla_close = np.roll(close_6h, 1)
    camarilla_range = camarilla_high - camarilla_low
    
    # Camarilla levels
    h5 = camarilla_close + camarilla_range * 1.1 / 2
    h4 = camarilla_close + camarilla_range * 1.1 / 4
    h3 = camarilla_close + camarilla_range * 1.1 / 6
    l3 = camarilla_close - camarilla_range * 1.1 / 6
    l4 = camarilla_close - camarilla_range * 1.1 / 4
    l5 = camarilla_close - camarilla_range * 1.1 / 2
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume average (24-period = 4 days on 6h)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=24, min_periods=24).mean().values
    
    # Align all indicators to 6h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_6h, h3)
    l3_aligned = align_htf_to_ltf(prices, df_6h, l3)
    h4_aligned = align_htf_to_ltf(prices, df_6h, h4)
    l4_aligned = align_htf_to_ltf(prices, df_6h, l4)
    h5_aligned = align_htf_to_ltf(prices, df_6h, h5)
    l5_aligned = align_htf_to_ltf(prices, df_6h, l5)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        h5_val = h5_aligned[i]
        l5_val = l5_aligned[i]
        ema50 = ema50_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above H3 with 1d uptrend and volume confirmation
            if price > h3_val and close_6h[i] > ema50 and vol > 1.5 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 with 1d downtrend and volume confirmation
            elif price < l3_val and close_6h[i] < ema50 and vol > 1.5 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to H-L range (between H4 and L4) OR volume drops
            if price < h4_val and price > l4_val:
                signals[i] = 0.0
                position = 0
            elif vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to H-L range (between H4 and L4) OR volume drops
            if price < h4_val and price > l4_val:
                signals[i] = 0.0
                position = 0
            elif vol < vol_ma:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0