#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter with Donchian breakout.
# Use weekly Choppiness Index to determine regime: >61.8 = range (mean revert), <38.2 = trending (breakout).
# In trending regime: long when price breaks above Donchian(20) high, short when breaks below Donchian(20) low.
# In ranging regime: long when price touches Donchian low, short when touches Donchian high.
# Volume confirmation: volume > 1.3x 20-period average.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "12h_Chop_Donchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Choppiness Index calculation
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (14-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # ATR (14-period Wilder's smoothing)
    def wilder_smooth(data, period):
        result = np.zeros_like(data)
        if len(data) < period:
            return result
        result[period-1] = np.nansum(data[:period])
        for i in range(period, len(data)):
            result[i] = result[i-1] - (result[i-1] / period) + data[i]
        return result
    
    atr_1w = wilder_smooth(tr, 14)
    # Sum of TR over 14 periods
    tr_sum_14 = wilder_smooth(tr, 14)  # Using same function for sum
    
    # Highest high and lowest low over 14 periods
    def highest_high(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period-1, len(arr)):
            result[i] = np.max(arr[i-period+1:i+1])
        return result
    
    def lowest_low(arr, period):
        result = np.full_like(arr, np.nan)
        for i in range(period-1, len(arr)):
            result[i] = np.min(arr[i-period+1:i+1])
        return result
    
    hh_14 = highest_high(high_1w, 14)
    ll_14 = lowest_low(low_1w, 14)
    
    # Avoid division by zero
    safe_tr_sum = np.where(tr_sum_14 == 0, np.finfo(float).eps, tr_sum_14)
    chop = 100 * np.log10(safe_tr_sum / (hh_14 - ll_14)) / np.log10(14)
    # Handle cases where hh_14 == ll_14
    chop = np.where((hh_14 - ll_14) == 0, 50, chop)  # Neutral when no range
    
    # Get daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high_1d, 20)
    donch_low = rolling_min(low_1d, 20)
    
    # Align indicators to 12h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Ensure Chop (14*2+6), Donchian (20), and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(chop_aligned[i]) or np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        chop_val = chop_aligned[i]
        donch_high_val = donch_high_aligned[i]
        donch_low_val = donch_low_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.3 * vol_ma
        
        # Regime determination
        is_ranging = chop_val > 61.8
        is_trending = chop_val < 38.2
        # Neutral zone (38.2-61.8) - no trades
        
        if position == 0:
            # Determine entry based on regime
            if is_trending and volume_confirmed:
                # Trending regime: breakout entries
                if price > donch_high_val:
                    signals[i] = 0.25
                    position = 1
                elif price < donch_low_val:
                    signals[i] = -0.25
                    position = -1
            elif is_ranging and volume_confirmed:
                # Ranging regime: mean reversion at extremes
                if price <= donch_low_val:
                    signals[i] = 0.25
                    position = 1
                elif price >= donch_high_val:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price crosses Donchian midline or opposite extreme based on regime
            midline = (donch_high_val + donch_low_val) / 2
            if is_trending:
                # In trending regime, exit when price crosses below midline
                if price < midline:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # In ranging regime, exit when price reaches opposite extreme
                if price >= donch_high_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses Donchian midline or opposite extreme based on regime
            midline = (donch_high_val + donch_low_val) / 2
            if is_trending:
                # In trending regime, exit when price crosses above midline
                if price > midline:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # In ranging regime, exit when price reaches opposite extreme
                if price <= donch_low_val:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals