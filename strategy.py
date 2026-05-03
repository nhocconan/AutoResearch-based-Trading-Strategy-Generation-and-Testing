#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d ADX trend filter and volume confirmation
# Long when price breaks above Donchian(20) high with volume > 1.5x 24-bar average and 1d ADX > 25 (strong trend)
# Short when price breaks below Donchian(20) low with volume > 1.5x 24-bar average and 1d ADX > 25 (strong trend)
# Exit when price crosses Donchian(20) midpoint or ADX < 20 (weak trend)
# Donchian channels provide objective breakout levels, ADX filters for trending markets, volume confirms conviction.
# Target: 75-200 total trades over 4 years = 19-50/year. Uses discrete sizing (0.30) to balance return and fees.

name = "4h_Donchian20_1dADX_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX for trend filter (using 14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = np.concatenate([[np.nan], high_1d[1:] - high_1d[:-1]])
    down_move = np.concatenate([[np.nan], low_1d[:-1] - low_1d[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilders_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = previous_smoothed - (previous_smoothed/period) + current
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
        return result
    
    atr_1d = wilders_smooth(tr, 14)
    plus_di_1d = 100 * wilders_smooth(plus_dm, 14) / atr_1d
    minus_di_1d = 100 * wilders_smooth(minus_dm, 14) / atr_1d
    dx_1d = 100 * np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d)
    adx_1d = wilders_smooth(dx_1d, 14)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian(20) channels on 4h data
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().shift(1).values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation (1.5x 24-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough for all calculations)
    start_idx = max(lookback + 1, 24 + 1, 14 * 3)  # Donchian(20) + volume MA(24) + ADX warmup
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high with volume spike and strong trend (ADX > 25)
            if (close[i] > donchian_high[i] and 
                volume_spike[i] and adx_1d_aligned[i] > 25):
                signals[i] = 0.30
                position = 1
            # Short entry: price breaks below Donchian low with volume spike and strong trend (ADX > 25)
            elif (close[i] < donchian_low[i] and 
                  volume_spike[i] and adx_1d_aligned[i] > 25):
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian mid or trend weakens (ADX < 20)
            if (close[i] < donchian_mid[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian mid or trend weakens (ADX < 20)
            if (close[i] > donchian_mid[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals