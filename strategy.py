#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation.
# In choppy markets (CHOP > 61.8), we fade moves; in trending markets (CHOP < 38.2), we follow breakouts.
# Uses 1d Donchian channels for breakout direction, volume confirmation for confirmation.
# Designed to work in both bull and bear markets by adapting to regime.
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 14-period Choppiness Index on 4h data
    def calculate_choppiness(high, low, close, period=14):
        atr = np.zeros(len(high))
        tr = np.zeros(len(high))
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Calculate ATR using Wilder's smoothing
        atr[period-1] = np.mean(tr[1:period])  # First ATR value
        for i in range(period, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        # Calculate sum of true ranges over period
        tr_sum = np.zeros(len(high))
        for i in range(period-1, len(high)):
            if i == period-1:
                tr_sum[i] = np.sum(tr[1:period])
            else:
                tr_sum[i] = tr_sum[i-1] - tr[i-period+1] + tr[i]
        
        # Calculate max/min close over period
        max_close = np.zeros(len(high))
        min_close = np.zeros(len(high))
        for i in range(len(high)):
            if i < period-1:
                max_close[i] = np.nan
                min_close[i] = np.nan
            else:
                max_close[i] = np.max(close[i-period+1:i+1])
                min_close[i] = np.min(close[i-period+1:i+1])
        
        # Calculate Choppiness Index
        chop = np.zeros(len(high))
        for i in range(period-1, len(high)):
            if max_close[i] > min_close[i] and tr_sum[i] > 0:
                chop[i] = 100 * np.log10(tr_sum[i] / (max_close[i] - min_close[i])) / np.log10(period)
            else:
                chop[i] = 50.0  # Default when no range
        return chop
    
    chop = calculate_choppiness(high, low, close, 14)
    
    # Calculate 20-period Donchian channels on daily data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    donchian_high = np.zeros(len(high_1d))
    donchian_low = np.zeros(len(high_1d))
    for i in range(len(high_1d)):
        if i < 19:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high_1d[i-19:i+1])
            donchian_low[i] = np.min(low_1d[i-19:i+1])
    
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate average volume (20-period) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(chop[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        chop_value = chop[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # In trending market (CHOP < 38.2): follow breakouts
            if chop_value < 38.2:
                # Long: price breaks above Donchian high with volume
                if price > upper and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short: price breaks below Donchian low with volume
                elif price < lower and volume_confirm:
                    position = -1
                    signals[i] = -position_size
            # In choppy market (CHOP > 61.8): fade moves (mean reversion)
            elif chop_value > 61.8:
                # Long: price near lower Donchian band with volume
                if price < lower * 1.02 and volume_confirm:
                    position = 1
                    signals[i] = position_size
                # Short: price near upper Donchian band with volume
                elif price > upper * 0.98 and volume_confirm:
                    position = -1
                    signals[i] = -position_size
            # In transition zone (38.2 <= CHOP <= 61.8): no trading
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches opposite Donchian band or chop becomes too high
            if (price >= upper * 0.995 or chop_value > 65.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches opposite Donchian band or chop becomes too high
            if (price <= lower * 1.005 or chop_value > 65.0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Chop_Donchian_Volume_Regime_v1"
timeframe = "4h"
leverage = 1.0