# 4H-1D CAMARILLA PIVOT STRATEGY V1
# Targets BTC/ETH/SOL with institutional pivot levels, volume confirmation, and regime filtering.
# Uses 1D Camarilla pivot levels (support/resistance) with 4H price action for entries.
# Includes volume spike confirmation and Choppiness Index regime filter to avoid false breakouts.
# Designed for low trade frequency (<50/year) with high win rate in both bull and bear markets.

#!/usr/bin/env python3
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
    
    # Get 1D data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: R4 = C + ((H-L) * 1.1/2), R3 = C + ((H-L) * 1.1/4), etc.
    # We focus on R3, R2, S3, S2 levels as key institutional levels
    hl_range = high_1d - low_1d
    camarilla_r3 = close_1d + (hl_range * 1.1 / 4)
    camarilla_r2 = close_1d + (hl_range * 1.1 / 6)
    camarilla_s2 = close_1d - (hl_range * 1.1 / 6)
    camarilla_s3 = close_1d - (hl_range * 1.1 / 4)
    
    # Align Camarilla levels to 4H timeframe (use previous day's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_r2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r2)
    camarilla_s2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s2)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Calculate 4H Choppiness Index for regime detection (14-period)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        atr = np.zeros(len(close_arr))
        true_range = np.zeros(len(close_arr))
        
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = np.abs(high_arr[i] - close_arr[i-1])
            lc = np.abs(low_arr[i] - close_arr[i-1])
            true_range[i] = max(hl, hc, lc)
        
        # Calculate ATR with smoothing
        atr[period] = np.mean(true_range[1:period+1])
        for i in range(period+1, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Calculate Chop
        chop = np.full(len(close_arr), 50.0)  # Default neutral
        for i in range(period, len(close_arr)):
            if atr[i] > 0:
                highest_high = np.max(high_arr[i-period+1:i+1])
                lowest_low = np.min(low_arr[i-period+1:i+1])
                if highest_high > lowest_low:
                    log_sum = np.sum(np.log(true_range[i-period+1:i+1] / atr[i]))
                    chop[i] = 100 - (100 * np.log(highest_high - lowest_low) / np.log(2) / log_sum)
        return chop
    
    chop = calculate_chop(high, low, close, 14)
    
    # Calculate volume spike detector (20-period average)
    volume_ma = np.zeros(len(volume))
    volume_sum = 0
    for i in range(len(volume)):
        volume_sum += volume[i]
        if i >= 20:
            volume_sum -= volume[i-20]
        if i < 20:
            volume_ma[i] = np.mean(volume[:i+1]) if i > 0 else volume[i]
        else:
            volume_ma[i] = volume_sum / 20
    
    volume_ratio = np.zeros(len(volume))
    for i in range(len(volume)):
        if volume_ma[i] > 0:
            volume_ratio[i] = volume[i] / volume_ma[i]
        else:
            volume_ratio[i] = 1.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):  # Start after warmup period
        # Skip if Camarilla data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_r2_aligned[i]) or 
            np.isnan(camarilla_s2_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: Only trade in trending markets (Chop < 61.8) or extreme ranges (Chop > 38.2)
        # We avoid choppy markets where Chop is between 38.2-61.8
        chop_value = chop[i] if not np.isnan(chop[i]) else 50.0
        is_trending_or_extreme = (chop_value < 38.2) or (chop_value > 61.8)
        
        # Volume confirmation: require volume spike (1.5x average)
        volume_spike = volume_ratio[i] > 1.5 if not np.isnan(volume_ratio[i]) else False
        
        # Price levels
        r3 = camarilla_r3_aligned[i]
        r2 = camarilla_r2_aligned[i]
        s2 = camarilla_s2_aligned[i]
        s3 = camarilla_s3_aligned[i]
        price = close[i]
        
        # Long setup: Price rejects S3/S2 support with volume
        long_setup = False
        if price <= s3 * 1.005:  # Near S3 support (0.5% tolerance)
            # Check for rejection: close above open AND volume spike
            if i > 0 and close[i] > prices['open'].iloc[i] and volume_spike:
                long_setup = True
        elif price <= s2 * 1.005:  # Near S2 support
            if i > 0 and close[i] > prices['open'].iloc[i] and volume_spike:
                long_setup = True
        
        # Short setup: Price rejects R3/R2 resistance with volume
        short_setup = False
        if price >= r3 * 0.995:  # Near R3 resistance (0.5% tolerance)
            # Check for rejection: close below open AND volume spike
            if i > 0 and close[i] < prices['open'].iloc[i] and volume_spike:
                short_setup = True
        elif price >= r2 * 0.995:  # Near R2 resistance
            if i > 0 and close[i] < prices['open'].iloc[i] and volume_spike:
                short_setup = True
        
        # Entry logic: Only in favorable regimes
        if long_setup and is_trending_or_extreme and position != 1:
            position = 1
            signals[i] = position_size
        elif short_setup and is_trending_or_extreme and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit logic: Opposite setup or loss of momentum
        elif position == 1 and (short_setup or (price < s2 and not is_trending_or_extreme)):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (long_setup or (price > r2 and not is_trending_or_extreme)):
            position = 0
            signals[i] = 0.0
        # Hold position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_pivot_v1"
timeframe = "4h"
leverage = 1.0