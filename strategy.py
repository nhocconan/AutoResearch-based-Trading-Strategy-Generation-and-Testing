#!/usr/bin/env python3
"""
6h_Donchian20_Breakout_1dADXTrend_VolumeConfirm
Hypothesis: Donchian(20) breakouts on 6h with 1d ADX(14)>25 trend filter and volume confirmation (1.8x 24-bar avg). 
Only trade breakouts aligned with strong 1d trend to avoid whipsaws in ranging markets. Volume confirms institutional participation.
Designed for 6h timeframe targeting 15-25 trades/year. Works in bull/bear by following 1d ADX trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF trend filter (ADX) and Donchian levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1d data for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    def wilder_smooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.nansum(data[:period])
            for i in range(period, len(data)):
                result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr = wilder_smooth(tr, 14)
    plus_di = 100 * wilder_smooth(plus_dm, 14) / atr
    minus_di = 100 * wilder_smooth(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx, 14)
    
    # Align ADX to 6h timeframe (1-day lagged for completed bar)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx, additional_delay_bars=1)
    
    # Calculate Donchian(20) levels on 6h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 1.8x 24-bar average volume (48h = 2 days on 6h)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.8 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for Donchian(20) and ADX
    start_idx = max(20, 50)  # 50 for ADX warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine 1d HTF trend: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        trend_bullish = strong_trend and (plus_di_aligned[i] > minus_di_aligned[i]) if 'plus_di_aligned' in locals() else False
        trend_bearish = strong_trend and (minus_di_aligned[i] > plus_di_aligned[i]) if 'minus_di_aligned' in locals() else False
        
        # Need to align +DI and -DI as well
        plus_di_aligned = align_htf_to_ltf(prices, df_1d, plus_di, additional_delay_bars=1)
        minus_di_aligned = align_htf_to_ltf(prices, df_1d, minus_di, additional_delay_bars=1)
        trend_bullish = strong_trend and (plus_di_aligned[i] > minus_di_aligned[i])
        trend_bearish = strong_trend and (minus_di_aligned[i] > plus_di_aligned[i])
        
        if position == 0:
            # Look for breakout signals with volume confirmation and strong trend alignment
            long_signal = (close[i] > highest_high[i]) and volume_spike[i] and trend_bullish
            short_signal = (close[i] < lowest_low[i]) and volume_spike[i] and trend_bearish
            
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
            # Exit when price breaks below lowest low or trend weakens (ADX < 20)
            exit_signal = (close[i] < lowest_low[i]) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit when price breaks above highest high or trend weakens (ADX < 20)
            exit_signal = (close[i] > highest_high[i]) or (adx_aligned[i] < 20)
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian20_Breakout_1dADXTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0