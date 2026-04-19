#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 12h ADX trend filter and volume confirmation
# Donchian provides clear breakout levels for trend capture
# 12h ADX > 25 ensures we only trade in trending markets, avoiding whipsaws in ranges
# Volume confirmation filters weak breakouts
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries
# Works in bull markets via breakouts, in bear markets via short breakdowns
name = "6h_Donchian20_12hADX_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h ADX for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate ADX on 12h data
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = 0.0
    tr2[0] = 0.0
    tr3[0] = 0.0
    tr_12h = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Plus Directional Movement (+DM) and Minus Directional Movement (-DM)
    up_move = high_12h - np.roll(high_12h, 1)
    down_move = np.roll(low_12h, 1) - low_12h
    up_move[0] = 0.0
    down_move[0] = 0.0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothing (Wilder's smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value is simple average
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
        return result
    
    period_adx = 14
    atr_12h = wilders_smoothing(tr_12h, period_adx)
    plus_di_12h = 100 * wilders_smoothing(plus_dm, period_adx) / atr_12h
    minus_di_12h = 100 * wilders_smoothing(minus_dm, period_adx) / atr_12h
    dx_12h = 100 * np.abs(plus_di_12h - minus_di_12h) / (plus_di_12h + minus_di_12h)
    adx_12h = wilders_smoothing(dx_12h, period_adx)
    
    # Align ADX to 6h timeframe (wait for completed 12h bar)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Volume confirmation: volume > 1.5 * 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, 20, period_adx*2)  # Ensure enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # ADX threshold for trending market
        trending = adx_12h_aligned[i] > 25
        
        if position == 0:
            # Long: Donchian breakout above + trending + volume confirmation
            if (close[i] > highest_high[i] and 
                trending and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below + trending + volume confirmation
            elif (close[i] < lowest_low[i] and 
                  trending and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if breakdown below Donchian lower or trend weakens
            if (close[i] < lowest_low[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if breakout above Donchian upper or trend weakens
            if (close[i] > highest_high[i]) or (adx_12h_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals