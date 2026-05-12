#!/usr/bin/env python3
# 6h_MarketProfile_ValueArea_Breakout_TrendVolume_v1
# Hypothesis: 6h price action trading using daily Market Profile value area (VA) with trend filter.
# Uses 1d Value Area High (VAH) and Value Area Low (VAL) from volume-weighted TPO calculation.
# Breakouts above VAH in uptrend (price > 200 EMA) or breakdowns below VAL in downtrend (price < 200 EMA)
# with volume confirmation (1.5x 20-period average) capture institutional activity.
# Works in bull/bear markets by following 200 EMA trend. Targets 15-25 trades/year on 6f timeframe.

name = "6h_MarketProfile_ValueArea_Breakout_TrendVolume_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)

    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values

    # Get 6h data for price action
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 2:
        return np.zeros(n)

    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values

    # Get 1d data for Market Profile Value Area calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:  # Need minimum for VA calculation
        return np.zeros(n)

    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate 200 EMA for trend filter (using 1d data aligned to 6h)
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)

    # Calculate Market Profile Value Area (70% of volume) for each day
    vah_1d = np.full(len(df_1d), np.nan)
    val_1d = np.full(len(df_1d), np.nan)
    poc_1d = np.full(len(df_1d), np.nan)  # Point of Control

    for i in range(len(df_1d)):
        # Get price range for the day
        day_low = low_1d[i]
        day_high = high_1d[i]
        if day_high <= day_low:
            continue
            
        # Create price profile using volume at each price level
        # Simplified: use volume distribution across price range
        price_bins = 50
        bin_width = (day_high - day_low) / price_bins
        if bin_width <= 0:
            continue
            
        price_levels = day_low + np.arange(price_bins) * bin_width
        volume_profile = np.zeros(price_bins)
        
        # Approximate volume distribution (using OHLC as proxy)
        # Assign volume to price levels based on typical price
        typical_price = (high_1d[i] + low_1d[i] + close_1d[i]) / 3.0
        bin_index = int((typical_price - day_low) / bin_width)
        if 0 <= bin_index < price_bins:
            volume_profile[bin_index] = volume_1d[i]
        
        # Find Point of Control (price with maximum volume)
        if np.sum(volume_profile) > 0:
            poc_index = np.argmax(volume_profile)
            poc_1d[i] = price_levels[poc_index]
            
            # Calculate Value Area (70% of volume around POC)
            total_volume = np.sum(volume_profile)
            target_volume = total_volume * 0.7
            
            # Expand out from POC to find VA boundaries
            volume_accumulated = volume_profile[poc_index]
            lower_index = poc_index
            upper_index = poc_index
            
            while volume_accumulated < target_volume:
                # Check volume at lower and upper levels
                volume_lower = volume_profile[lower_index - 1] if lower_index > 0 else 0
                volume_upper = volume_profile[upper_index + 1] if upper_index < price_bins - 1 else 0
                
                # Expand towards the side with more volume
                if volume_lower >= volume_upper and lower_index > 0:
                    lower_index -= 1
                    volume_accumulated += volume_profile[lower_index]
                elif volume_upper > volume_lower and upper_index < price_bins - 1:
                    upper_index += 1
                    volume_accumulated += volume_profile[upper_index]
                else:
                    # Can't expand further, break
                    break
            
            val_1d[i] = price_levels[lower_index]
            vah_1d[i] = price_levels[upper_index]

    # Align VA levels to 6h timeframe
    vah_1d_aligned = align_htf_to_ltf(prices, df_1d, vah_1d)
    val_1d_aligned = align_htf_to_ltf(prices, df_1d, val_1d)

    # Calculate 6h volume SMA20 for volume confirmation
    volume_series = pd.Series(volume_6h)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike_threshold = volume_sma20 * 1.5  # Require 1.5x average volume

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(200, n):  # Start after EMA200 warmup
        # Skip if any required data is NaN
        if (np.isnan(vah_1d_aligned[i]) or np.isnan(val_1d_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(volume_sma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Breakout above VAH in uptrend with volume spike
            if (close[i] > vah_1d_aligned[i] and 
                close[i] > ema200_1d_aligned[i] and 
                volume[i] > volume_sma20[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Breakdown below VAL in downtrend with volume spike
            elif (close[i] < val_1d_aligned[i] and 
                  close[i] < ema200_1d_aligned[i] and 
                  volume[i] > volume_sma20[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes back into Value Area or trend reversal
            if close[i] < vah_1d_aligned[i] or close[i] < ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes back into Value Area or trend reversal
            if close[i] > val_1d_aligned[i] or close[i] > ema200_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals