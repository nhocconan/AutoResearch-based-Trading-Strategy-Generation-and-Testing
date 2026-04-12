#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Camarilla H3/L3 breakout with 12h EMA50 trend filter and volume confirmation
    # Uses 12h Camarilla levels (H3/L3 for breakouts) and 12h EMA50 for trend
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Designed for low trade frequency (target: 20-40/year) to minimize fee drag
    # Trend filter works in bull/bear markets; breakout structure captures momentum
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels and EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate 12h Camarilla pivot levels (H3/L3 for breakouts)
    camarilla_h3_12h = np.full(len(df_12h), np.nan)
    camarilla_l3_12h = np.full(len(df_12h), np.nan)
    pivot_12h = np.full(len(df_12h), np.nan)
    
    for i in range(1, len(df_12h)):
        high_val = high_12h[i-1]
        low_val = low_12h[i-1]
        close_val = close_12h[i-1]
        pivot_val = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        pivot_12h[i] = pivot_val
        camarilla_h3_12h[i] = pivot_val + range_val * 1.1 / 4.0  # H3
        camarilla_l3_12h[i] = pivot_val - range_val * 1.1 / 4.0  # L3
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_h3_12h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_l3_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (4h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Determine 12h trend
        bullish_trend = close[i] > ema50_12h_aligned[i]
        bearish_trend = close[i] < ema50_12h_aligned[i]
        
        # Entry logic: Camarilla breakout with volume and trend filter
        long_entry = False
        short_entry = False
        
        # Long breakout: price breaks above camarilla H3 in bullish trend with volume
        if bullish_trend:
            long_entry = (close[i] > camarilla_h3_aligned[i]) and volume_spike[i]
        # Short breakout: price breaks below camarilla L3 in bearish trend with volume
        elif bearish_trend:
            short_entry = (close[i] < camarilla_l3_aligned[i]) and volume_spike[i]
        
        # Exit logic: opposite camarilla level or trend reversal
        long_exit = bearish_trend and close[i] < camarilla_l3_aligned[i]
        short_exit = bullish_trend and close[i] > camarilla_h3_aligned[i]
        
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

name = "4h_12h_camarilla_h3l3_ema50_volume_v1"
timeframe = "4h"
leverage = 1.0