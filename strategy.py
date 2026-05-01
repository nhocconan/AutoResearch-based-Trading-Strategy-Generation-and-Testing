#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d weekly pivot bias and volume confirmation
# Uses 1d data to compute weekly pivot points (H+L+C)/3 from prior completed week
# Long when price > weekly pivot and breaks above Donchian(20) upper band with volume spike
# Short when price < weekly pivot and breaks below Donchian(20) lower band with volume spike
# Weekly pivot acts as dynamic support/resistance that adapts to both bull and bear markets
# Volume confirmation > 1.8x 20-period EMA filters low-quality breakouts
# Designed for low trade frequency: ~20-40 trades/year per symbol with 0.25 sizing
# Weekly pivot bias ensures trades are taken in direction of higher timeframe structure

name = "4h_Donchian20_WeeklyPivot_Bias_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for weekly pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly OHLC from daily data using 7-day lookback
    # Weekly high = max of prior 7 daily highs
    # Weekly low = min of prior 7 daily lows  
    # Weekly close = close of 7th prior day
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().shift(7).values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().shift(7).values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).apply(lambda x: x[-1], raw=True).shift(7).values
    
    # Weekly pivot point: (HIGH + LOW + CLOSE) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    
    # 4h Donchian(20) breakout
    donchian_window = 20
    donchian_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().shift(1).values
    
    # Volume confirmation: volume > 1.8 * 20-period EMA
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup: need 1d data for weekly pivot (7*2=14 days min) + Donchian20 + volume EMA20
    # 14 days * 6 (4h bars per day) = 84 bars
    start_idx = max(14*6, donchian_window, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ema_20[i])):
            signals[i] = 0.0
            continue
        
        # Determine bias from weekly pivot: long above pivot, short below pivot
        bullish_bias = close[i] > weekly_pivot_aligned[i]
        bearish_bias = close[i] < weekly_pivot_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if bullish_bias:
                # Long: Donchian breakout above upper band with volume spike
                if close[i] > donchian_high[i] and volume_spike[i]:
                    signals[i] = 0.25
                    position = 1
                else:
                    signals[i] = 0.0
            elif bearish_bias:
                # Short: Donchian breakdown below lower band with volume spike
                if close[i] < donchian_low[i] and volume_spike[i]:
                    signals[i] = -0.25
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0  # Avoid chop around pivot
        
        elif position == 1:  # Long position
            # Exit: Donchian breakdown below lower band (failure of breakout)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Donchian breakout above upper band (failure of breakdown)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals