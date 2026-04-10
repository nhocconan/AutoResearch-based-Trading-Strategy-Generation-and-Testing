#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# - Long: price breaks above Donchian(20) high + 1d ADX > 25 (trending) + 4h volume > 1.5x 20-period MA
# - Short: price breaks below Donchian(20) low + 1d ADX > 25 (trending) + 4h volume > 1.5x 20-period MA
# - Exit: price returns to Donchian(20) midpoint or opposite breakout
# - Position sizing: 0.30 (discrete level)
# - Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and fee drag
# - Donchian channels provide objective breakout levels; ADX filters for trending markets to avoid whipsaws
# - Volume confirmation ensures breakout validity, reducing false signals

name = "4h_1d_donchian_breakout_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:  # Need enough data for calculations
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 4h OHLCV
    open_4h = prices['open'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    close_4h = prices['close'].values
    volume_4h = prices['volume'].values
    
    # Pre-compute 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels for 4h
    highest_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Calculate 1d ADX(14) for trend filter
    # ADX requires +DI, -DI, and TR
    tr1 = pd.Series(high_1d).rolling(2).max().values - pd.Series(low_1d).rolling(2).min().values
    tr2 = abs(pd.Series(high_1d).rolling(2).shift(1).values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr3 = abs(pd.Series(low_1d).rolling(2).shift(1).values - pd.Series(close_1d).rolling(2).shift(1).values)
    tr = np.maximum.reduce([tr1, tr2, tr3])
    tr[0] = high_1d[0] - low_1d[0]  # First value
    
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    plus_dm = np.insert(plus_dm, 0, 0)
    
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    minus_dm = np.insert(minus_dm, 0, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (equivalent to EMA with alpha=1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h volume moving average (20-period)
    volume_ma_20_4h = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup period (need at least 20 for Donchian)
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_ma_20_4h[i])):
            signals[i] = 0.0
            continue
        
        # Get current 4h data
        close_price = close_4h[i]
        volume_current = volume_4h[i]
        
        # Get aligned 1d data for current 4h bar (completed 1d bar)
        adx_current = adx_aligned[i]
        
        # Trend condition: ADX > 25 indicates trending market
        trending = adx_current > 25
        
        # Volume spike condition: current 4h volume > 1.5x 20-period MA
        volume_spike = volume_current > 1.5 * volume_ma_20_4h[i]
        
        # Breakout conditions
        breakout_up = close_price > highest_high[i]  # Break above Donchian high
        breakout_down = close_price < lowest_low[i]   # Break below Donchian low
        
        if position == 0:  # Flat - look for new entries
            # Long entry: breakout above Donchian high + trending + volume spike
            if breakout_up and trending and volume_spike:
                position = 1
                signals[i] = 0.30
            # Short entry: breakout below Donchian low + trending + volume spike
            elif breakout_down and trending and volume_spike:
                position = -1
                signals[i] = -0.30
            else:
                signals[i] = 0.0
        else:  # Have position - look for exit
            # Exit when price returns to Donchian midpoint or opposite breakout
            if position == 1:  # Long position
                if close_price <= donchian_mid[i] or breakout_down:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.30
            else:  # position == -1 (short position)
                if close_price >= donchian_mid[i] or breakout_up:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.30
    
    return signals