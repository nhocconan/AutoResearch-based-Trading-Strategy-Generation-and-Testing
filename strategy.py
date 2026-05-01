#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above Donchian upper AND weekly pivot > previous weekly pivot (bullish bias) AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower AND weekly pivot < previous weekly pivot (bearish bias) AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to minimize fee churn. Weekly pivot provides structural bias from higher timeframe.
# Donchian channels capture breakouts from consolidation, effective in both trending and ranging markets.
# Volume confirmation reduces false breakouts. Designed for 6h timeframe to target 12-37 trades/year.

name = "6h_Donchian20_WeeklyPivot_VolumeSpike_v1"
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
    open_time = prices['open_time']
    
    # Pre-compute session hours for efficiency
    hours = pd.DatetimeIndex(open_time).hour
    
    # Load weekly data ONCE before loop for pivot bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard: (H+L+C)/3)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    weekly_pivot = (high_1w + low_1w + close_1w) / 3.0
    
    # Align weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    
    # Weekly pivot bias: rising pivot = bullish bias, falling pivot = bearish bias
    weekly_pivot_slope = np.diff(weekly_pivot_aligned, prepend=weekly_pivot_aligned[0])
    weekly_pivot_rising = weekly_pivot_slope > 0  # bullish bias
    weekly_pivot_falling = weekly_pivot_slope < 0  # bearish bias
    
    # Donchian channels (20-period)
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Volume confirmation: current 6h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)  # warmup for Donchian and volume MA
    
    for i in range(start_idx, n):
        # Session filter: trade all sessions for 6h timeframe
        hour = hours[i]
        
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)
        
        # Donchian breakout signals
        breakout_up = curr_high > donchian_high[i]  # break above upper band
        breakout_down = curr_low < donchian_low[i]  # break below lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout above Donchian upper AND weekly pivot rising (bullish bias) AND volume confirmation
            if (breakout_up and 
                weekly_pivot_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian lower AND weekly pivot falling (bearish bias) AND volume confirmation
            elif (breakout_down and 
                  weekly_pivot_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price crosses below Donchian lower (stoploss) OR weekly pivot falls (bias change)
            if (curr_low < donchian_low[i] or 
                weekly_pivot_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price crosses above Donchian upper (stoploss) OR weekly pivot rises (bias change)
            if (curr_high > donchian_high[i] or 
                weekly_pivot_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals