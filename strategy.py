#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 1-day ATR filter and 1-day volume confirmation.
Trades breakouts of 20-period Donchian channel when ATR-based volatility is rising and volume exceeds 1-day average.
Designed to work in both bull and bear markets by using volatility expansion as a filter and volume to confirm breakout strength.
Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.
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
    
    # Get 4-hour data for Donchian calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4-hour Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Upper and lower bands
    upper = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4-hour timeframe
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    
    # Get daily data for ATR filter and volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1-day ATR(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Calculate 1-day volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Warmup: need Donchian channels, ATR, and volume MA
    start_idx = max(20, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        atr_now = atr_aligned[i]
        
        # Current Donchian levels
        upper_now = upper_aligned[i]
        lower_now = lower_aligned[i]
        
        # Volatility filter: ATR rising (current ATR > ATR 3 periods ago)
        vol_filter = atr_now > atr_aligned[i-3] if i >= 3 else False
        
        # Volume filter: volume > 1.2x 1-day average
        vol_confirm = vol_now > 1.2 * vol_ma
        
        # Entry conditions: Donchian breakout with volatility expansion and volume confirmation
        if position == 0:
            # Long: break above upper band with volatility expansion and volume
            if price_now > upper_now and vol_filter and vol_confirm:
                signals[i] = size
                position = 1
            # Short: break below lower band with volatility expansion and volume
            elif price_now < lower_now and vol_filter and vol_confirm:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower band or volatility drops
            if price_now < lower_now or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper band or volatility drops
            if price_now > upper_now or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_DonchianBreakout_ATRVol_VolumeConfirm"
timeframe = "4h"
leverage = 1.0