#!/usr/bin/env python3
"""
1d_WeeklyDonchian20_Breakout_WeeklyTrend_ChopFilter_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and choppiness regime to capture medium-term trends while avoiding whipsaws in ranging markets.
- Uses 1d timeframe targeting 30-100 total trades over 4 years (7-25/year)
- Donchian breakout from previous 20-day high/low (structure-based entry)
- Weekly EMA50 trend filter to ensure alignment with higher timeframe momentum
- Weekly choppiness filter (CHOP < 38.2) to avoid false breakouts in ranging markets
- Volume confirmation (1.5x 20-day average volume) to increase signal reliability
- Designed for low trade frequency to minimize fee drag while maintaining edge in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend and chop filters
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate weekly choppiness index to filter ranging markets
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_arr = df_1w['close'].values
    
    # True Range calculation
    tr1 = np.maximum(high_1w - low_1w, np.absolute(high_1w - np.roll(close_1w_arr, 1)))
    tr2 = np.maximum(np.absolute(low_1w - np.roll(close_1w_arr, 1)), tr1)
    tr2[0] = high_1w[0] - low_1w[0]  # First TR
    atr14_1w = pd.Series(tr2).rolling(window=14, min_periods=14).mean().values
    
    highest_high_14_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_14_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range_14_1w = highest_high_14_1w - lowest_low_14_1w
    hl_range_14_1w = np.where(hl_range_14_1w == 0, 1e-10, hl_range_14_1w)
    
    chop_1w = 100 * np.log10(atr14_1w * 14 / np.log10(14) / hl_range_14_1w) / np.log10(100)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Calculate Donchian channels (20-period) on daily timeframe
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation (1.5x 20-day average volume)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for Donchian/volume, 50 for weekly EMA, 14 for weekly chop)
    start_idx = max(20, 50, 14)
    
    for i in range(start_idx, n):
        # Skip if any weekly data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Donchian breakout conditions
        price_above_donchian_high = high[i] > highest_high_20[i-1]  # Break above previous 20-day high
        price_below_donchian_low = low[i] < lowest_low_20[i-1]      # Break below previous 20-day low
        
        # Weekly trend filter
        trend_up = close[i] > ema50_1w_aligned[i]
        trend_down = close[i] < ema50_1w_aligned[i]
        
        # Weekly choppiness filter: only trade when market is trending (CHOP < 38.2)
        trending_market = chop_1w_aligned[i] < 38.2
        
        if position == 0:
            # Long: price breaks above Donchian high AND volume spike AND weekly uptrend AND trending market
            if price_above_donchian_high and volume_spike[i] and trend_up and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND volume spike AND weekly downtrend AND trending market
            elif price_below_donchian_low and volume_spike[i] and trend_down and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price falls below Donchian low OR weekly trend turns down OR market becomes choppy
            if low[i] < lowest_low_20[i] or not trend_up or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price rises above Donchian high OR weekly trend turns up OR market becomes choppy
            if high[i] > highest_high_20[i] or not trend_down or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyDonchian20_Breakout_WeeklyTrend_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0