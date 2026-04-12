#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA50 trend filter and volume confirmation
    # Uses 4h Camarilla levels (H3/L3 for breakouts) and 4h EMA50 for trend
    # Volume spike (>2.0x 20-period average) confirms institutional participation
    # Designed for low trade frequency (target: 15-37/year) to minimize fee drag
    # Trend filter works in bull/bear markets; breakout structure captures momentum
    # Session filter (08-20 UTC) reduces noise trades
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels and EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate 4h Camarilla pivot levels (H3/L3 for breakouts/continuation)
    camarilla_h3_4h = np.full(len(df_4h), np.nan)
    camarilla_l3_4h = np.full(len(df_4h), np.nan)
    pivot_4h = np.full(len(df_4h), np.nan)
    
    for i in range(1, len(df_4h)):
        high_val = high_4h[i-1]
        low_val = low_4h[i-1]
        close_val = close_4h[i-1]
        pivot_val = (high_val + low_val + close_val) / 3.0
        range_val = high_val - low_val
        
        pivot_4h[i] = pivot_val
        camarilla_h3_4h[i] = pivot_val + range_val * 1.1 / 4.0  # H3
        camarilla_l3_4h[i] = pivot_val - range_val * 1.1 / 4.0  # L3
    
    # Calculate 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h3_4h)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l3_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: volume > 2.0 * 20-period average (1h)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Session filter: 08-20 UTC (pre-compute hours array)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_h3_aligned[i]) or 
            np.isnan(camarilla_l3_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Determine 4h trend
        bullish_trend = close[i] > ema50_4h_aligned[i]
        bearish_trend = close[i] < ema50_4h_aligned[i]
        
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
            signals[i] = 0.20
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.20
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_4h_camarilla_h3l3_ema50_volume_v1"
timeframe = "1h"
leverage = 1.0