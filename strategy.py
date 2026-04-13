#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d volume spike + ADX regime filter.
    # Donchian breakout captures momentum in trending markets.
    # Volume spike confirms institutional participation.
    # ADX > 25 ensures we only trade in trending regimes (avoids chop).
    # Target: 50-150 total trades over 4 years = 12-37/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF indicators (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 1d ADX (14-period) for regime filter
    # +DM, -DM, TR calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr = np.concatenate([[np.nan], tr])  # first value NaN
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[np.nan], plus_dm])
    minus_dm = np.concatenate([[np.nan], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        # Subsequent values: Wilder's smoothing
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI+ and DI-
    plus_di14 = np.where(tr14 != 0, (plus_dm14 / tr14) * 100, 0)
    minus_di14 = np.where(tr14 != 0, (minus_dm14 / tr14) * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di14 + minus_di14) != 0, 
                  np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14) * 100, 0)
    adx = wilders_smoothing(dx, 14)
    
    # Calculate 1d volume MA(20) for spike detection
    volume_1d = df_1d['volume'].values
    volume_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_std = pd.Series(volume_1d).rolling(window=20, min_periods=20).std().values
    volume_threshold = volume_ma + (2.0 * volume_std)  # 2-sigma spike
    
    # Align HTF indicators to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    volume_threshold_aligned = align_htf_to_ltf(prices, df_1d, volume_threshold)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_threshold_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1d volume threshold (spike)
        # Need to get corresponding 1d volume for this 12h bar
        # Since we aligned the 1d threshold, we can use it directly
        volume_filter = volume[i] > volume_threshold_aligned[i]
        
        # Regime filter: ADX > 25 indicates trending market
        regime_filter = adx_aligned[i] > 25
        
        # Donchian breakout conditions
        breakout_long = close[i] > donchian_high_aligned[i]  # Break above upper channel
        breakout_short = close[i] < donchian_low_aligned[i]  # Break below lower channel
        
        # Entry conditions: breakout with volume spike and in trending regime
        long_entry = breakout_long and volume_filter and regime_filter
        short_entry = breakout_short and volume_filter and regime_filter
        
        # Exit conditions: price returns to opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.30
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.30
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_volume_adx_regime_v1"
timeframe = "12h"
leverage = 1.0