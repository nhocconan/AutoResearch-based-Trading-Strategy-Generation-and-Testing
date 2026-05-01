#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout + 1d ADX regime filter + volume confirmation
# Uses 1d ADX(14) to define regime: ADX>25 = trending (trade breakouts), ADX<20 = range (fade to mean)
# Donchian(20) breakout provides clean entry/exit with low trade frequency
# Volume confirmation (1.5x average) filters weak breakouts
# Designed for low frequency (75-200 trades over 4 years) with clear bull/bear logic

name = "4h_Donchian20_1dADX_Regime_Volume_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # 1d ADX(14) calculation for regime detection
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Wilder's smoothing
    def wilders_smoothing(x, period):
        result = np.full_like(x, np.nan)
        if len(x) >= period:
            first_val = np.nansum(x[1:period+1])
            result[period] = first_val
            for i in range(period+1, len(x)):
                result[i] = result[i-1] - (result[i-1] / period) + x[i]
        return result
    
    tr_period = 14
    tr_smoothed = wilders_smoothing(tr, tr_period)
    dm_plus_smoothed = wilders_smoothing(dm_plus, tr_period)
    dm_minus_smoothed = wilders_smoothing(dm_minus, tr_period)
    
    # DI+ and DI-
    di_plus = np.where(tr_smoothed != 0, (dm_plus_smoothed / tr_smoothed) * 100, 0)
    di_minus = np.where(tr_smoothed != 0, (dm_minus_smoothed / tr_smoothed) * 100, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = wilders_smoothing(dx, tr_period)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, 34)  # Need Donchian and ADX
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Regime filters
        trending = adx_aligned[i] > 25
        ranging = adx_aligned[i] < 20
        
        if position == 0:  # Flat - look for new entries
            # Volume confirmation
            vol_confirmed = volume[i] > 1.5 * vol_ma[i]
            
            if vol_confirmed:
                # Trending regime: Donchian breakout
                if trending:
                    # Long: price breaks above upper Donchian channel
                    if close[i] > highest_high[i-1]:
                        signals[i] = 0.25
                        position = 1
                    # Short: price breaks below lower Donchian channel
                    elif close[i] < lowest_low[i-1]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                # Ranging regime: fade Donchian extremes (mean reversion)
                elif ranging:
                    # Long: price touches or breaks below lower Donchian channel
                    if close[i] <= lowest_low[i-1]:
                        signals[i] = 0.25
                        position = 1
                    # Short: price touches or breaks above upper Donchian channel
                    elif close[i] >= highest_high[i-1]:
                        signals[i] = -0.25
                        position = -1
                    else:
                        signals[i] = 0.0
                else:
                    signals[i] = 0.0  # Transition regime (ADX 20-25) - stay flat
            else:
                signals[i] = 0.0  # No volume confirmation
        
        elif position == 1:  # Long position
            # Exit conditions
            exit_long = False
            if trending:
                # Exit trending long when price breaks below lower Donchian channel
                if close[i] < lowest_low[i-1]:
                    exit_long = True
            elif ranging:
                # Exit ranging long when price reaches upper Donchian channel (mean reversion target)
                if close[i] >= highest_high[i-1]:
                    exit_long = True
            else:
                # Transition regime - exit on Donchian middle line
                middle = (highest_high[i-1] + lowest_low[i-1]) / 2
                if close[i] <= middle:
                    exit_long = True
            
            if exit_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            if trending:
                # Exit trending short when price breaks above upper Donchian channel
                if close[i] > highest_high[i-1]:
                    exit_short = True
            elif ranging:
                # Exit ranging short when price reaches lower Donchian channel (mean reversion target)
                if close[i] <= lowest_low[i-1]:
                    exit_short = True
            else:
                # Transition regime - exit on Donchian middle line
                middle = (highest_high[i-1] + lowest_low[i-1]) / 2
                if close[i] >= middle:
                    exit_short = True
            
            if exit_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals