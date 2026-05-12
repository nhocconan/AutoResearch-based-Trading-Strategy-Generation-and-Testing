#!/usr/bin/env python3
# 6H_Market_Profile_Value_Area_Pullback_1wTrend_Filter
# Hypothesis: Buy pullbacks to 1-day Value Area High (VAH) in uptrend and sell rallies to Value Area Low (VAL) in downtrend using weekly trend filter. Market Profile identifies institutional value zones; price tends to respect these areas before continuing with higher timeframe trend. Works in bull by buying dips to VAH, works in bear by selling rallies to VAL. Uses 6h timeframe with weekly trend filter for higher timeframe context.

name = "6H_Market_Profile_Value_Area_Pullback_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)

    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values

    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')

    # Calculate weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)

    # Get daily data for Market Profile calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values

    # Calculate Value Area (70% of volume) using volume-weighted price bins
    vah = np.full(len(close_1d), np.nan)
    val = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i < 20:  # Need minimum lookback for VA calculation
            continue
            
        # Get last 20 days of data
        start_idx = max(0, i - 19)
        end_idx = i + 1
        
        if end_idx - start_idx < 5:  # Need minimum data
            continue
            
        # Create price bins
        price_min = np.min(low_1d[start_idx:end_idx])
        price_max = np.max(high_1d[start_idx:end_idx])
        if price_max <= price_min:
            continue
            
        n_bins = 50
        bin_size = (price_max - price_min) / n_bins
        if bin_size <= 0:
            continue
            
        # Volume profile
        volume_profile = np.zeros(n_bins)
        
        for j in range(start_idx, end_idx):
            # Distribute volume across the day's range
            bin_low = int((low_1d[j] - price_min) / bin_size)
            bin_high = int((high_1d[j] - price_min) / bin_size)
            bin_low = max(0, min(bin_low, n_bins - 1))
            bin_high = max(0, min(bin_high, n_bins - 1))
            
            if bin_high >= bin_low:
                vol_per_bin = volume_1d[j] / (bin_high - bin_low + 1)
                for k in range(bin_low, bin_high + 1):
                    volume_profile[k] += vol_per_bin
        
        # Find Value Area (70% of volume)
        total_volume = np.sum(volume_profile)
        if total_volume <= 0:
            continue
            
        target_volume = 0.7 * total_volume
        
        # Find point of control (max volume bin)
        poc_bin = np.argmax(volume_profile)
        poc_price = price_min + poc_bin * bin_size
        
        # Build value area around POC
        volume_accum = volume_profile[poc_bin]
        bin_low = poc_bin
        bin_high = poc_bin
        
        while volume_accum < target_volume and (bin_low > 0 or bin_high < n_bins - 1):
            # Expand to the side with more volume
            vol_down = volume_profile[bin_low - 1] if bin_low > 0 else 0
            vol_up = volume_profile[bin_high + 1] if bin_high < n_bins - 1 else 0
            
            if vol_down >= vol_up and bin_low > 0:
                bin_low -= 1
                volume_accum += volume_profile[bin_low]
            elif bin_high < n_bins - 1:
                bin_high += 1
                volume_accum += volume_profile[bin_high]
            else:
                break
        
        # Calculate VAH and VAL
        vah[i] = price_min + bin_high * bin_size
        val[i] = price_min + bin_low * bin_size

    # Align Value Area levels to 6h timeframe
    vah_aligned = align_htf_to_ltf(prices, df_1d, vah)
    val_aligned = align_htf_to_ltf(prices, df_1d, val)

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):  # Start after warmup
        if (np.isnan(vah_aligned[i]) or np.isnan(val_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Pullback to VAH in weekly uptrend
            if (close[i] <= vah_aligned[i] * 1.005 and  # Within 0.5% of VAH
                close[i] > val_aligned[i] and          # Above VAL
                close[i] > ema_50_1w_aligned[i]):      # Weekly uptrend
                signals[i] = 0.25
                position = 1
            # SHORT: Rally to VAL in weekly downtrend
            elif (close[i] >= val_aligned[i] * 0.995 and  # Within 0.5% of VAL
                  close[i] < vah_aligned[i] and           # Below VAH
                  close[i] < ema_50_1w_aligned[i]):       # Weekly downtrend
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes above VAH (breakout) or below VAL (breakdown)
            if close[i] > vah_aligned[i] * 1.01 or close[i] < val_aligned[i] * 0.99:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes below VAL (breakdown) or above VAH (breakout)
            if close[i] < val_aligned[i] * 0.99 or close[i] > vah_aligned[i] * 1.01:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals