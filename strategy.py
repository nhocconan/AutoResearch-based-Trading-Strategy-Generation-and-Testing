#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly pivot direction and volume confirmation.
# Long when price breaks above 6h Donchian upper band AND weekly pivot trend is up AND volume > 1.5x 20-period median.
# Short when price breaks below 6h Donchian lower band AND weekly pivot trend is down AND volume > 1.5x 20-period median.
# Weekly pivot trend provides higher-timeframe bias (bull/bear), Donchian captures breakouts, volume confirms strength.
# Works in bull markets (buy breakouts in weekly uptrend) and bear markets (sell breakdowns in weekly downtrend).
# Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag.

name = "6h_Donchian20_Breakout_WeeklyPivot_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 6h Donchian channels (20-period)
    donchian_window = 20
    upper_band = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lower_band = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate 1w OHLC for weekly pivot trend (using prior week's close)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly pivot trend: based on prior week's close vs open
    # Uptrend if weekly close > weekly open, downtrend if weekly close < weekly open
    prev_week_open = df_1w['open'].values
    prev_week_close = df_1w['close'].values
    weekly_trend_up = prev_week_close > prev_week_open
    weekly_trend_down = prev_week_close < prev_week_open
    
    # Align weekly trend to 6h timeframe (use prior week's trend)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up.astype(float))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down.astype(float))
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0  # track entry price for stoploss
    
    # Start after warmup for Donchian and volume
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or 
            np.isnan(weekly_trend_up_aligned[i]) or 
            np.isnan(weekly_trend_down_aligned[i]) or 
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above upper band AND weekly trend up AND volume spike
            if curr_high > upper_band[i] and weekly_trend_up_aligned[i] > 0.5 and volume_confirm:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            # Short: price breaks below lower band AND weekly trend down AND volume spike
            elif curr_low < lower_band[i] and weekly_trend_down_aligned[i] > 0.5 and volume_confirm:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls back below upper band OR weekly trend turns down
            if curr_close < upper_band[i] or weekly_trend_up_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises back above lower band OR weekly trend turns up
            if curr_close > lower_band[i] or weekly_trend_down_aligned[i] <= 0.5:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -0.25
    
    return signals