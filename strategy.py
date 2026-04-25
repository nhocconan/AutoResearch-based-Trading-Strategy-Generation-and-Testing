#!/usr/bin/env python3
"""
6h_ADX_Regime_Donchian_Breakout
Hypothesis: On 6h timeframe, use 1d ADX to filter regimes (ADX>25 = trending, ADX<20 = ranging). 
In trending regime: trade Donchian(20) breakouts in direction of 1d EMA50 trend. 
In ranging regime: fade Donchian(20) breakouts (mean reversion at channel extremes). 
Volume confirmation (>1.5x 20-bar avg) required for all entries. 
Designed for low trade frequency (~20-40/year) to work in both bull and bear markets via regime adaptation.
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
    
    # Get 1d data for HTF regime and trend filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate ADX(14) on 1d for regime filter
    # ADX requires +DI, -DI, and DX calculation
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[np.nan], plus_dm])
        minus_dm = np.concatenate([[np.nan], minus_dm])
        
        # Smoothed TR, +DM, -DM (Wilder's smoothing)
        def wilder_smooth(data, period):
            result = np.full_like(data, np.nan)
            if len(data) < period:
                return result
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: smoothed = prev_smoothed - (prev_smoothed/period) + current
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = result[i-1] - (result[i-1]/period) + data[i]
            return result
        
        tr_smooth = wilder_smooth(tr, period)
        plus_dm_smooth = wilder_smooth(plus_dm, period)
        minus_dm_smooth = wilder_smooth(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(tr_smooth != 0, 100 * plus_dm_smooth / tr_smooth, 0)
        minus_di = np.where(tr_smooth != 0, 100 * minus_dm_smooth / tr_smooth, 0)
        
        # DX and ADX
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilder_smooth(dx, period)
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate EMA50 on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Donchian(20) channels on 6h data
    def donchian_channels(high, low, period=20):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
    # Align HTF indicators to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx_1d, additional_delay_bars=1)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d, additional_delay_bars=1)
    
    # Volume confirmation: 1.5x 20-bar average volume
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ADX (~14*3=42), EMA50 (50), Donchian (20)
    start_idx = max(50, 42)  # EMA50 needs 50 bars
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(ema50_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        if position == 0:
            # Regime-based breakout logic
            # Trending regime: ADX > 25 -> breakout continuation
            # Ranging regime: ADX < 20 -> breakout fade (mean reversion)
            adx_val = adx_aligned[i]
            
            if adx_val > 25:  # Trending regime
                # Long: price breaks above Donchian upper in uptrend (close > EMA50) with volume spike
                # Short: price breaks below Donchian lower in downtrend (close < EMA50) with volume spike
                long_signal = (close[i] > donchian_upper[i]) and (close[i] > ema50_aligned[i]) and volume_spike[i]
                short_signal = (close[i] < donchian_lower[i]) and (close[i] < ema50_aligned[i]) and volume_spike[i]
            elif adx_val < 20:  # Ranging regime
                # Long: price breaks below Donchian lower (fakeout) -> mean reversion long
                # Short: price breaks above Donchian upper (fakeout) -> mean reversion short
                long_signal = (close[i] < donchian_lower[i]) and volume_spike[i]
                short_signal = (close[i] > donchian_upper[i]) and volume_spike[i]
            else:  # Transition regime (20 <= ADX <= 25) -> no trade
                long_signal = False
                short_signal = False
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit when price moves back below Donchian middle or opposite signal
            exit_signal = close[i] < (donchian_upper[i] + donchian_lower[i]) / 2
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price moves back above Donchian middle or opposite signal
            exit_signal = close[i] > (donchian_upper[i] + donchian_lower[i]) / 2
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ADX_Regime_Donchian_Breakout"
timeframe = "6h"
leverage = 1.0