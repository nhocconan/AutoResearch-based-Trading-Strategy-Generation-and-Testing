#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1w Donchian breakout with volume confirmation.
# Uses weekly Donchian channels (20-period) for breakout signals, filtered by 12h Choppiness Index
# to avoid whipsaw in ranging markets (CHOP > 61.8) and only trade in strong trends (CHOP < 38.2).
# Volume confirmation ensures breakouts have conviction. Designed to work in both bull and bear
# markets by capturing strong trending moves while avoiding choppy periods.
# Weekly timeframe provides fewer, higher-quality signals suitable for 12h chart.
name = "12h_ChopFilter_1wDonchian_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Donchian channels (20-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels: 20-period high/low
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels using rolling window
    donchian_high = np.full_like(high_1w, np.nan)
    donchian_low = np.full_like(low_1w, np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_1w[i-19:i+1])
            donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align Donchian levels to 12h timeframe (wait for weekly bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # 12h data for Choppiness Index
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 14:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_12h[1:] - low_12h[1:])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    # Prepend first TR as high-low for first period
    tr = np.concatenate([[high_12h[0] - low_12h[0]], tr])
    
    # ATR (14-period)
    atr = np.full_like(close_12h, np.nan)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])  # First ATR is simple average
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14  # Wilder's smoothing
    
    # Sum of ATR over 14 periods
    atr_sum = np.full_like(close_12h, np.nan)
    for i in range(len(atr_sum)):
        if i >= 13 and not np.isnan(atr[i]):
            # Sum of last 14 ATR values
            start_idx = max(0, i - 13)
            atr_sum[i] = np.sum(atr[start_idx:i+1])
    
    # Choppiness Index: 100 * log10(ATR_sum / (highest_high - lowest_low)) / log10(14)
    highest_high = np.full_like(close_12h, np.nan)
    lowest_low = np.full_like(close_12h, np.nan)
    
    for i in range(len(high_12h)):
        if i >= 13:  # 14-period lookback
            highest_high[i] = np.max(high_12h[i-13:i+1])
            lowest_low[i] = np.min(low_12h[i-13:i+1])
    
    chop = np.full_like(close_12h, np.nan)
    for i in range(len(chop)):
        if i >= 13 and not np.isnan(atr_sum[i]) and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(atr_sum[i] / (highest_high[i] - lowest_low[i])) / np.log10(14)
    
    # Align Choppiness Index to 12h timeframe (no extra delay needed as it's based on completed 12h bar)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume confirmation: volume > 1.5x 20-period EMA on 12h timeframe
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ema20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(chop_aligned[i]) or np.isnan(vol_ema20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + trend regime (CHOP < 38.2) + volume
            if (price > donchian_high_aligned[i] and 
                chop_aligned[i] < 38.2 and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + trend regime (CHOP < 38.2) + volume
            elif (price < donchian_low_aligned[i] and 
                  chop_aligned[i] < 38.2 and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below weekly Donchian low OR choppy regime (CHOP > 61.8)
            if price < donchian_low_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above weekly Donchian high OR choppy regime (CHOP > 61.8)
            if price > donchian_high_aligned[i] or chop_aligned[i] > 61.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals