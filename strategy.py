#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + volume confirmation + ATR stoploss.
Long when price breaks above Donchian upper band with volume > 1.5x average volume.
Short when price breaks below Donchian lower band with volume > 1.5x average volume.
Exit via ATR-based stoploss (2.0 ATR from entry) or opposite Donchian break.
Uses 1w for trend filter (price > 1w EMA50 for longs, price < 1w EMA50 for shorts).
Target: 50-150 total trades over 4 years (12-37/year). Donchian captures breakouts,
volume confirmation filters false breakouts, weekly EMA50 ensures trend alignment.
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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate Donchian channels (20-period) on 12h
    high_12h_series = pd.Series(high_12h)
    low_12h_series = pd.Series(low_12h)
    donchian_upper = high_12h_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_12h_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian channels to primary timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_lower)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema50_1w = close_1w_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to primary timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate ATR (14) for stoploss on primary timeframe
    high_s = pd.Series(high)
    low_s = pd.Series(low)
    close_s = pd.Series(close)
    tr1 = high_s - low_s
    tr2 = abs(high_s - close_s.shift(1))
    tr3 = abs(low_s - close_s.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate average volume (20-period) for volume confirmation
    volume_s = pd.Series(volume)
    avg_volume = volume_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_at_entry = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema50 = ema50_1w_aligned[i]
        atr_val = atr[i]
        avg_vol = avg_volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper + volume > 1.5x avg + price > 1w EMA50
            if price > upper and vol > 1.5 * avg_vol and price > ema50:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_at_entry = atr_val
            # Short: price breaks below Donchian lower + volume > 1.5x avg + price < 1w EMA50
            elif price < lower and vol > 1.5 * avg_vol and price < ema50:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_at_entry = atr_val
        
        elif position == 1:
            # Exit long: ATR stoploss (2.0 ATR below entry) OR price < Donchian lower
            if price <= entry_price - 2.0 * atr_at_entry or price < lower:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: ATR stoploss (2.0 ATR above entry) OR price > Donchian upper
            if price >= entry_price + 2.0 * atr_at_entry or price > upper:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_VolumeConfirm_1wEMA50_Trend_ATRStop"
timeframe = "12h"
leverage = 1.0