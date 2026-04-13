#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 1d Donchian(20) breakout + 1w HMA(21) trend filter + volume confirmation
    # Long: Price breaks above Donchian(20) high AND 1w HMA(21) rising AND volume > 1.5x avg
    # Short: Price breaks below Donchian(20) low AND 1w HMA(21) falling AND volume > 1.5x avg
    # Exit: Opposite Donchian breakout OR HMA trend reversal
    # Using 1d for price action/volume, 1w for trend filter to avoid whipsaw
    # Discrete position sizing (0.25) to balance return and drawdown
    # Target: 20-50 trades/year (~80-200 over 4 years) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels and volume (call ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for HMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1d Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian high: rolling max of high
    donch_high_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Donchian low: rolling min of low
    donch_low_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d average volume (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1w HMA(21) for trend filter
    close_1w = df_1w['close'].values
    
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(data, window):
        """Weighted Moving Average"""
        if len(data) < window:
            return np.full_like(data, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(data, weights / weights.sum(), mode='valid')
    
    # Calculate WMA for half period
    half_period = 21 // 2  # 10
    wma_half = wma(close_1w, half_period)
    # Pad to match length
    wma_half_padded = np.full_like(close_1w, np.nan)
    wma_half_padded[half_period-1:len(wma_half)+half_period-1] = wma_half
    
    # Calculate WMA for full period
    wma_full = wma(close_1w, 21)
    wma_full_padded = np.full_like(close_1w, np.nan)
    wma_full_padded[21-1:len(wma_full)+21-1] = wma_full
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half_padded - wma_full_padded
    # Final HMA: WMA(sqrt(n)) of raw HMA
    sqrt_period = int(np.sqrt(21))  # 4
    hma_1w = wma(raw_hma, sqrt_period)
    hma_1w_padded = np.full_like(close_1w, np.nan)
    hma_1w_padded[sqrt_period-1:len(hma_1w)+sqrt_period-1] = hma_1w
    
    # Align 1d indicators to 1d timeframe (no alignment needed as we're already on 1d)
    # Align 1w HMA to 1d timeframe (wait for completed 1w bar)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_padded)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high_1d[i]) or np.isnan(donch_low_1d[i]) or 
            np.isnan(avg_vol_1d[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Get aligned 1d values (since we're on 1d timeframe, use current values)
        # For 1d timeframe, we need to align the 1d indicators to themselves
        # Actually, since we're using 1d as primary timeframe, we can use the 1d arrays directly
        # But we need to ensure we're using completed 1d bar data
        # The get_htf_data for 1d already gives us completed 1d bars
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_high_1d[i-1]  # break above previous Donchian high
        breakout_down = close[i] < donch_low_1d[i-1]  # break below previous Donchian low
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * avg_vol_1d[i]
        
        # HMA trend filter: check if HMA is rising or falling
        # Need previous HMA value to determine direction
        if i > 0 and not np.isnan(hma_1w_aligned[i-1]):
            hma_rising = hma_1w_aligned[i] > hma_1w_aligned[i-1]
            hma_falling = hma_1w_aligned[i] < hma_1w_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # Entry logic
        long_entry = breakout_up and hma_rising and volume_spike
        short_entry = breakout_down and hma_falling and volume_spike
        
        # Exit logic: opposite breakout OR HMA trend reversal
        long_exit = breakout_down or (hma_falling and not hma_rising)  # opposite breakout or trend turning down
        short_exit = breakout_up or (hma_rising and not hma_falling)   # opposite breakout or trend turning up
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_donchian_hma_volume_v1"
timeframe = "1d"
leverage = 1.0