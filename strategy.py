#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d trend filter (price above/below Kumo cloud) and volume confirmation.
- Primary timeframe: 4h targeting 75-200 total trades over 4 years (19-50/year).
- HTF: 1d for trend filter (Ichimoku Kumo cloud) to avoid counter-trend trades.
- Entry: Long when price breaks above Donchian(20) high AND bullish 1d trend AND volume > 1.5x avg volume.
         Short when price breaks below Donchian(20) low AND bearish 1d trend AND volume > 1.5x avg volume.
- Exit: Opposite Donchian breakout OR price crosses Kumo in opposite direction.
- Signal size: 0.25 discrete to minimize fee drag.
- Donchian provides clear structure, Kumo filter avoids bad regimes, volume confirms conviction.
- Works in bull markets (buy breakouts above cloud) and bear markets (sell breakdowns below cloud).
- Estimated trades: ~100 total over 4 years (~25/year) based on breakout frequency with filters.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ichimoku(high, low, close, tenkan=9, kijun=26, senkou=52):
    """Calculate Ichimoku components."""
    # Tenkan-sen (Conversion Line): (9-period high + 9-period low)/2
    period9_high = pd.Series(high).rolling(window=tenkan, min_periods=tenkan).max().values
    period9_low = pd.Series(low).rolling(window=tenkan, min_periods=tenkan).min().values
    tenkan_sen = (period9_high + period9_low) / 2
    
    # Kijun-sen (Base Line): (26-period high + 26-period low)/2
    period26_high = pd.Series(high).rolling(window=kijun, min_periods=kijun).max().values
    period26_low = pd.Series(low).rolling(window=kijun, min_periods=kijun).min().values
    kijun_sen = (period26_high + period26_low) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan-sen + Kijun-sen)/2
    senkou_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B): (52-period high + 52-period low)/2
    period52_high = pd.Series(high).rolling(window=senkou, min_periods=senkou).max().values
    period52_low = pd.Series(low).rolling(window=senkou, min_periods=senkou).min().values
    senkou_b = ((period52_high + period52_low) / 2)
    
    return tenkan_sen, kijun_sen, senkou_a, senkou_b

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d trend filter: Ichimoku Kumo cloud
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate Ichimoku on 1d data for trend filter
    tenkan_1d, kijun_1d, senkou_a_1d, senkou_b_1d = calculate_ichimoku(
        df_1d['high'].values, df_1d['low'].values, df_1d['close'].values
    )
    
    # Kumo cloud boundaries (Senkou Span A and B)
    kumo_top_1d = np.maximum(senkou_a_1d, senkou_b_1d)
    kumo_bottom_1d = np.minimum(senkou_a_1d, senkou_b_1d)
    
    # Trend filter: price above cloud = bullish, price below cloud = bearish
    close_1d = df_1d['close'].values
    bullish_trend = close_1d > kumo_top_1d
    bearish_trend = close_1d < kumo_bottom_1d
    
    # Align 1d indicators to 4h timeframe (wait for completed 1d candle + 1 bar for confirmation)
    bullish_trend_aligned = align_htf_to_ltf(prices, df_1d, bullish_trend, additional_delay_bars=1)
    bearish_trend_aligned = align_htf_to_ltf(prices, df_1d, bearish_trend, additional_delay_bars=1)
    kumo_top_aligned = align_htf_to_ltf(prices, df_1d, kumo_top_1d, additional_delay_bars=1)
    kumo_bottom_aligned = align_htf_to_ltf(prices, df_1d, kumo_bottom_1d, additional_delay_bars=1)
    
    # Calculate Donchian channels on 4h data (20-period)
    period = 20
    donchian_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    donchian_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Donchian breakout signals
    breakout_above = (close > donchian_high) & (np.roll(close, 1) <= np.roll(donchian_high, 1))
    breakout_below = (close < donchian_low) & (np.roll(close, 1) >= np.roll(donchian_low, 1))
    
    # Handle first element for roll
    breakout_above[0] = False
    breakout_below[0] = False
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 60  # Need sufficient data for Ichimoku (max period 52) and Donchian (20)
    
    for i in range(start_idx, n):
        # Skip if data not ready (check for NaN from alignment or calculations)
        if (np.isnan(bullish_trend_aligned[i]) or np.isnan(bearish_trend_aligned[i]) or 
            np.isnan(kumo_top_aligned[i]) or np.isnan(kumo_bottom_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(avg_volume[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        
        # Exit conditions: opposite Donchian breakout OR price crosses Kumo in opposite direction
        if position != 0:
            # Exit long: breakout below OR price falls below Kumo bottom
            if position == 1:
                if breakout_below[i] or curr_close < kumo_bottom_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
            # Exit short: breakout above OR price rises above Kumo top
            elif position == -1:
                if breakout_above[i] or curr_close > kumo_top_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    continue
        
        # Entry conditions: Donchian breakout in direction of 1d trend filter with volume confirmation
        if position == 0:
            # Long: breakout above AND bullish 1d trend AND volume confirmation
            if breakout_above[i] and bullish_trend_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: breakout below AND bearish 1d trend AND volume confirmation
            elif breakout_below[i] and bearish_trend_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: maintain signal
            signals[i] = 0.25
        elif position == -1:
            # Short position: maintain signal
            signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1dKomoTrend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0