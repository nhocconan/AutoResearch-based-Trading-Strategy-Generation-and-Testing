#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1w EMA200 AND volume > 1.5x 20-period average.
Short when price breaks below Donchian(20) low AND price < 1w EMA200 AND volume > 1.5x 20-period average.
Exit when price touches opposite Donchian(10) level (profit target) or trailing stop of 2x ATR(20).
Uses 1d for price action/volume/Donchian, 1w for EMA200 trend filter.
Target: 30-100 total trades over 4 years (7-25/year). Donchian provides clear breakout structure,
weekly EMA200 filters for higher-timeframe trend alignment to avoid counter-trend trades.
Volume confirmation ensures breakouts have conviction, reducing false signals.
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
    
    # Get 1d data for Donchian, volume, ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20 for breakout, 10 for exit)
    high_series = pd.Series(high_1d)
    low_series = pd.Series(low_1d)
    donchian_high_20 = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_series.rolling(window=20, min_periods=20).min().values
    donchian_high_10 = high_series.rolling(window=10, min_periods=10).max().values
    donchian_low_10 = low_series.rolling(window=10, min_periods=10).min().values
    
    # Calculate 1d ATR(20) for trailing stop
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d volume 20-period average
    volume_ma20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w EMA200 for trend filter
    close_1w_series = pd.Series(close_1w)
    ema200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 1d timeframe
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # Align 1d indicators to lower timeframe (if needed) - but we're using 1d as primary
    # For 1d timeframe, we can use values directly
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(donchian_high_20[i]) or np.isnan(atr[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        dh20 = donchian_high_20[i]
        dl20 = donchian_low_20[i]
        dh10 = donchian_high_10[i]
        dl10 = donchian_low_10[i]
        vol_ma = volume_ma20[i]
        atr_val = atr[i]
        ema200 = ema200_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = vol > 1.5 * vol_ma if not np.isnan(vol_ma) else False
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND price > 1w EMA200 AND volume confirmed
            if price > dh20 and price > ema200 and vol_confirmed:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Donchian(20) low AND price < 1w EMA200 AND volume confirmed
            elif price < dl20 and price < ema200 and vol_confirmed:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price touches Donchian(10) low OR trailing stop hit (price < highest_high - 2*ATR)
            # Track highest high since entry for trailing stop
            if i == start_idx:
                highest_since_entry = high[i]
            else:
                highest_since_entry = max(high[i], highest_since_entry) if 'highest_since_entry' in locals() else high[i]
            
            trailing_stop = highest_since_entry - 2.0 * atr_val
            
            if price < dl10 or price < trailing_stop:
                signals[i] = 0.0
                position = 0
                if 'highest_since_entry' in locals():
                    del highest_since_entry
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches Donchian(10) high OR trailing stop hit (price > lowest_low + 2*ATR)
            # Track lowest low since entry for trailing stop
            if i == start_idx:
                lowest_since_entry = low[i]
            else:
                lowest_since_entry = min(low[i], lowest_since_entry) if 'lowest_since_entry' in locals() else low[i]
            
            trailing_stop = lowest_since_entry + 2.0 * atr_val
            
            if price > dh10 or price > trailing_stop:
                signals[i] = 0.0
                position = 0
                if 'lowest_since_entry' in locals():
                    del lowest_since_entry
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_DonchianBreakout_Volume_WeeklyEMA200_Trend"
timeframe = "1d"
leverage = 1.0