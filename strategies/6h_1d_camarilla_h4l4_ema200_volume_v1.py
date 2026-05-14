#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Camarilla H4/L4 breakout with 1d EMA200 trend filter and volume confirmation
    # Uses daily Camarilla levels (H4/L4 for breakouts) and daily EMA200 for trend
    # Volume spike (>2.0x 50-period average) confirms institutional participation
    # Designed for low trade frequency (target: 15-30/year) to minimize fee drag
    # Trend filter works in bull/bear markets; breakout structure captures momentum
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot levels and EMA200
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Camarilla pivot levels (H4/L4 for breakouts/continuation)
    camarilla_h4_1d = np.full(len(df_1d), np.nan)
    camarilla_l4_1d = np.full(len(df_1d), np.nan)
    pivot_1d = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        high_val = high_1d[i-1]
        low_val = low_1d[i-1]
        close_val = close_1d[i-1]
        pivot_val = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        pivot_1d[i] = pivot_val
        camarilla_h4_1d[i] = pivot_val + range_val * 1.1 / 2.0  # H4
        camarilla_l4_1d[i] = pivot_val - range_val * 1.1 / 2.0  # L4
    
    # Calculate 1d EMA200 for trend filter
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align HTF indicators to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4_1d)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 2.0 * 50-period average (6h)
    vol_ma = np.full(n, np.nan)
    for i in range(50, n):
        vol_ma[i] = np.mean(volume[i-50:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(camarilla_h4_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 1d trend
        bullish_trend = close[i] > ema200_1d_aligned[i]
        bearish_trend = close[i] < ema200_1d_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above camarilla H4 in bullish trend with volume
        if bullish_trend:
            long_entry = (close[i] > camarilla_h4_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below camarilla L4 in bearish trend with volume
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l4_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite camarilla level or trend reversal
        long_exit = bearish_trend and close[i] < camarilla_l4_aligned[i]
        short_exit = bullish_trend and close[i] > camarilla_h4_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_camarilla_h4l4_ema200_volume_v1"
timeframe = "6h"
leverage = 1.0