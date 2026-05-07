#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour price action with 1-day Choppiness Index regime filter.
# Uses 12h Donchian(20) breakout for entry, filtered by 1d Choppiness Index > 61.8 (ranging market) for mean reversion.
# Long when price breaks below Donchian Lower (20) in chop regime, short when breaks above Donchian Upper (20) in chop regime.
# Exit when price returns to Donchian Middle (20-period average of upper/lower).
# Designed for low frequency (12-25 trades/year) to avoid fee drag in ranging markets.
# Works in both bull and bear by capturing mean reversion in choppy conditions.
name = "12h_DonchianChoppiness_MeanReversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 12h Donchian channels (20-period)
    donchian_window = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    middle = np.full(n, np.nan)
    
    for i in range(donchian_window - 1, n):
        upper[i] = np.max(high[i - donchian_window + 1:i + 1])
        lower[i] = np.min(low[i - donchian_window + 1:i + 1])
        middle[i] = (upper[i] + lower[i]) / 2
    
    # Load 1d data for Choppiness Index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range (TR)
    tr = np.full(len(close_1d), np.nan)
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    # Average True Range (ATR) - 14 period
    atr_14 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        atr_14[13] = np.mean(tr[1:15])  # Simple average of first 14 TR values
        for i in range(14, len(close_1d)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Choppiness Index (CHOP) - 14 period
    chop = np.full(len(close_1d), np.nan)
    chop_period = 14
    if len(close_1d) >= chop_period + 1:
        for i in range(chop_period, len(close_1d)):
            # Sum of ATR over period
            sum_atr = np.sum(atr_14[i - chop_period + 1:i + 1])
            # Max high - min low over period
            max_high = np.max(high_1d[i - chop_period + 1:i + 1])
            min_low = np.min(low_1d[i - chop_period + 1:i + 1])
            if sum_atr > 0 and (max_high - min_low) > 0:
                chop[i] = 100 * np.log10(sum_atr / (max_high - min_low)) / np.log10(chop_period)
    
    # Align Chop to 12h timeframe (needs 2-bar delay for confirmation)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop, additional_delay_bars=2)
    
    # Define chop regime: CHOP > 61.8 = ranging (good for mean reversion)
    chop_regime = chop_aligned > 61.8
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, donchian_window)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter mean reversion in chop regime
            # Long when price breaks below lower band
            long_condition = (close[i] < lower[i]) and chop_regime[i]
            # Short when price breaks above upper band
            short_condition = (close[i] > upper[i]) and chop_regime[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price returns to middle
            if close[i] > middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price returns to middle
            if close[i] < middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals