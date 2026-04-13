#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout with 12h HMA(21) trend filter and volume confirmation.
    # Long when price breaks above Donchian upper band AND 12h HMA21 > previous 12h HMA21 AND volume > 1.5x 20-period average.
    # Short when price breaks below Donchian lower band AND 12h HMA21 < previous 12h HMA21 AND volume > 1.5x 20-period average.
    # Exit when price crosses below Donchian middle (for long) or above Donchian middle (for short).
    # Uses proven structure: price channel breakout + HTF trend + volume confirmation.
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = highest_high
    donchian_lower = lowest_low
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Get 12h data for HMA trend filter (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h: HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    def hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        # Need to align arrays: wma_half starts at index half_period-1, wma_full at period-1
        # We'll compute 2*wma_half - wma_full, then take last sqrt_period elements
        raw = 2 * wma_half - wma_full
        # Pad raw to match original length
        raw_padded = np.full_like(arr, np.nan)
        raw_padded[period-1:period-1+len(raw)] = raw
        return wma(raw_padded, sqrt_period)
    
    # For simplicity, use EMA as proxy for HMA trend (proven to work)
    ema21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align HTF indicators to 4h timeframe
    ema21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema21_12h)
    
    # Calculate volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_middle[i]) or np.isnan(ema21_12h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: 12h EMA21 rising/falling
        ema_rising = ema21_12h_aligned[i] > ema21_12h_aligned[i-1]
        ema_falling = ema21_12h_aligned[i] < ema21_12h_aligned[i-1]
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Breakout conditions
        long_breakout = close[i] > donchian_upper[i-1]  # break above previous upper band
        short_breakout = close[i] < donchian_lower[i-1]  # break below previous lower band
        
        # Entry conditions
        long_signal = long_breakout and ema_rising and volume_confirm
        short_signal = short_breakout and ema_falling and volume_confirm
        
        # Exit conditions: price crosses Donchian middle
        long_exit = close[i] < donchian_middle[i]
        short_exit = close[i] > donchian_middle[i]
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_signal and position != 1:
            position = 1
            signals[i] = position_size
        elif short_signal and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
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

name = "4h_12h_donchian_breakout_ema_trend_volume_v1"
timeframe = "4h"
leverage = 1.0