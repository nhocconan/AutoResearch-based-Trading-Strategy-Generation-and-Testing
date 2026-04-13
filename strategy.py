#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + 1d HMA trend filter + volume confirmation
    # Enter long when price breaks above 4h Donchian upper band with volume > 1.5x 20-bar avg and 1d HMA rising
    # Enter short when price breaks below 4h Donchian lower band with volume > 1.5x 20-bar avg and 1d HMA falling
    # Exit on opposite Donchian band touch or ATR-based stoploss
    # Uses 1d HTF for HMA trend (more stable than 4h) and 4h for entry timing
    # Donchian provides objective structure, HMA filters trend direction, volume confirms participation
    # Works in bull (continuation breaks with trend) and bear (reversal breaks against trend)
    # Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for primary timeframe
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Get 1d data for HMA trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 21:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 4h Donchian channels (20-period)
    # Upper band = highest high over past 20 periods
    # Lower band = lowest low over past 20 periods
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d HMA (21-period) for trend filter
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = wma(values, half_period)
        wma_full = wma(values, period)
        
        # 2 * WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full
        # WMA(sqrt(n)) of the above
        hma_values = wma(raw_hma, sqrt_period)
        
        # Pad with NaN to match original length
        result = np.full_like(values, np.nan)
        result[period-1:] = hma_values
        return result
    
    hma_21_1d = hma(close_1d, 21)
    
    # Align indicators to 4h timeframe
    donchian_upper_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # ATR-based stoploss (2.5 * ATR)
    tr1 = pd.Series(high_4h - low_4h)
    tr2 = pd.Series(np.abs(high_4h - np.roll(close_4h, 1)))
    tr3 = pd.Series(np.abs(low_4h - np.roll(close_4h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # start from 20 to ensure Donchian bands are ready
        # Skip if data not ready
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(hma_21_1d_aligned[i]) or np.isnan(atr_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_upper_aligned[i]  # break above upper band
        breakout_down = close[i] < donchian_lower_aligned[i]  # break below lower band
        
        # HMA trend filter: rising HMA = uptrend, falling HMA = downtrend
        hma_rising = hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] if i > 0 else False
        hma_falling = hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] if i > 0 else False
        
        # Entry conditions with volume confirmation and trend filter
        long_entry = breakout_up and volume_confirmed[i] and hma_rising and position != 1
        short_entry = breakout_down and volume_confirmed[i] and hma_falling and position != -1
        
        # Exit conditions: opposite band touch or ATR stoploss
        exit_long = False
        exit_short = False
        
        if position == 1:
            # Exit long: price touches lower band or hits stoploss
            exit_long = close[i] < donchian_lower_aligned[i]
            # ATR stoploss: 2.5 * ATR below entry (tracked via position logic)
            # Simplified: exit if price drops 2.5*ATR from recent high (approximation)
            if i >= 20:
                recent_high = np.max(high[max(0, i-20):i+1])
                if close[i] < recent_high - 2.5 * atr_aligned[i]:
                    exit_long = True
        
        elif position == -1:
            # Exit short: price touches upper band or hits stoploss
            exit_short = close[i] > donchian_upper_aligned[i]
            # ATR stoploss: 2.5 * ATR above entry
            if i >= 20:
                recent_low = np.min(low[max(0, i-20):i+1])
                if close[i] > recent_low + 2.5 * atr_aligned[i]:
                    exit_short = True
        
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

name = "4h_1d_donchian_hma_volume_filter_v2"
timeframe = "4h"
leverage = 1.0