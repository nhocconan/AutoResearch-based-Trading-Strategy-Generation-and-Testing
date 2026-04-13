#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Camarilla pivot breakout with 1d trend filter and volume confirmation
    # Long when: price breaks above 12h Donchian(20) high AND price > 1d Camarilla H3 AND volume > 1.3x avg volume AND 1d close > 1d EMA50
    # Short when: price breaks below 12h Donchian(20) low AND price < 1d Camarilla L3 AND volume > 1.3x avg volume AND 1d close < 1d EMA50
    # Exit when: price crosses 12h Donchian midpoint OR volume < avg volume
    # Uses discrete sizing (0.25) targeting 50-150 trades over 4 years.
    # Works in bull/bear via 1d EMA50 trend filter and Camarilla pivot structure providing dynamic S/R.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 1d Camarilla pivots (using previous day's range)
    range_1d = high_1d - low_1d
    h3_1d = close_1d + 1.125 * range_1d
    l3_1d = close_1d - 1.125 * range_1d
    h4_1d = close_1d + 1.5 * range_1d
    l4_1d = close_1d - 1.5 * range_1d
    
    # Align 1d indicators to 12h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    h4_1d_aligned = align_htf_to_ltf(prices, df_1d, h4_1d)
    l4_1d_aligned = align_htf_to_ltf(prices, df_1d, l4_1d)
    
    # Calculate Donchian(20) channels on 12h
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.3
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(h3_1d_aligned[i]) or
            np.isnan(l3_1d_aligned[i]) or np.isnan(vol_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_ok = volume[i] > vol_threshold[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Trend and Camarilla filters
        long_filter = (close[i] > h3_1d_aligned[i]) and (close_1d_aligned := ema50_1d_aligned[i]) and (close[i] > ema50_1d_aligned[i])
        short_filter = (close[i] < l3_1d_aligned[i]) and (close[i] < ema50_1d_aligned[i])
        
        # Entry conditions
        long_entry = long_breakout and long_filter and vol_ok and position != 1
        short_entry = short_breakout and short_filter and vol_ok and position != -1
        
        # Exit conditions: price crosses Donchian midpoint OR volume drops below average
        exit_long = close[i] < donchian_mid[i] or volume[i] < vol_ma[i]
        exit_short = close[i] > donchian_mid[i] or volume[i] < vol_ma[i]
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_camarilla_breakout_trend_volume_v1"
timeframe = "12h"
leverage = 1.0