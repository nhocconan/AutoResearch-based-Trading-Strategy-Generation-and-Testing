#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_Breakout_Volume
Hypothesis: On 6h timeframe, use 1d ADX to filter regime (ADX>25 = trending, ADX<20 = ranging).
In trending regime (ADX>25): trade Donchian(20) breakouts with volume confirmation.
In ranging regime (ADX<20): fade Donchian(20) touches at bands with volume confirmation.
Volume confirmation: current volume > 1.5 * 20-period average.
Position size: 0.25 discrete levels.
Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets.
Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate ADX(14) on 1d timeframe
    # TR = max(high-low, abs(high-close_prev), abs(low-close_prev))
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed TR, +DM, -DM (Wilder smoothing = EMA with alpha=1/period)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        alpha = 1.0 / period
        # First value: simple average
        if len(data) >= period:
            result[period-1] = np.nanmean(data[:period])
        # Wilder smoothing: today = (1-1/period)*yesterday + (1/period)*today
        for i in range(period, len(data)):
            if not np.isnan(result[i-1]):
                result[i] = result[i-1] * (1 - alpha) + data[i] * alpha
        return result
    
    atr_1d = wilders_smoothing(tr, 14)
    plus_dm_1d = wilders_smoothing(plus_dm, 14)
    minus_dm_1d = wilders_smoothing(minus_dm, 14)
    
    # +DI and -DI
    plus_di_1d = np.where(atr_1d != 0, (plus_dm_1d / atr_1d) * 100, 0)
    minus_di_1d = np.where(atr_1d != 0, (minus_dm_1d / atr_1d) * 100, 0)
    
    # DX and ADX
    dx_1d = np.where((plus_di_1d + minus_di_1d) != 0, 
                     np.abs(plus_di_1d - minus_di_1d) / (plus_di_1d + minus_di_1d) * 100, 0)
    adx_1d = wilders_smoothing(dx_1d, 14)
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate Donchian channels (20-period) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need ADX (14+14=28), Donchian (20), volume avg (20)
    start_idx = max(28, 20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        adx_val = adx_1d_aligned[i]
        upper_band = donchian_high[i]
        lower_band = donchian_low[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Regime-based entry logic
            if adx_val > 25:  # Trending regime - breakout
                # Long breakout: price closes above upper Donchian band
                long_breakout = (close_val > upper_band) and vol_conf
                # Short breakout: price closes below lower Donchian band
                short_breakout = (close_val < lower_band) and vol_conf
                
                if long_breakout:
                    signals[i] = size
                    position = 1
                elif short_breakout:
                    signals[i] = -size
                    position = -1
                    
            elif adx_val < 20:  # Ranging regime - mean reversion at bands
                # Long when price touches lower band and reverses up
                long_reversion = (low_val <= lower_band) and (close_val > lower_band) and vol_conf
                # Short when price touches upper band and reverses down
                short_reversion = (high_val >= upper_band) and (close_val < upper_band) and vol_conf
                
                if long_reversion:
                    signals[i] = size
                    position = 1
                elif short_reversion:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: regime change or opposite signal
            exit_condition = (adx_val < 20) or (close_val < lower_band)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: regime change or opposite signal
            exit_condition = (adx_val < 20) or (close_val > upper_band)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ADX_Regime_Donchian_Breakout_Volume"
timeframe = "6h"
leverage = 1.0