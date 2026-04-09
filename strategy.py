#!/usr/bin/env python3
# 1d_1w_volatility_breakout_v1
# Hypothesis: Daily breakout above 1-week Donchian high/low with ATR-based position sizing.
# Long when close > weekly Donchian high + volatility filter (ATR ratio > 1.2).
# Short when close < weekly Donchian low + volatility filter (ATR ratio > 1.2).
# Exit when price crosses weekly midpoint (mean of high/low).
# Position size scaled by ATR volatility (0.2-0.3 range) to manage drawdown in bear markets.
# Works in bull via breakout continuation and in bear via volatility filtering to avoid false breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_volatility_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 20-period weekly Donchian channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    donchian_high = np.full(len(high_1w), np.nan)
    donchian_low = np.full(len(low_1w), np.nan)
    
    for i in range(len(high_1w)):
        if i >= 19:  # 20-period lookback
            donchian_high[i] = np.max(high_1w[i-19:i+1])
            donchian_low[i] = np.min(low_1w[i-19:i+1])
    
    # Align Donchian levels to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    
    # Calculate weekly midpoint for exit
    donchian_mid = (donchian_high + donchian_low) / 2
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1w, donchian_mid)
    
    # Calculate daily ATR(14) for volatility filter and position sizing
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr = np.full(n, np.nan)
    if n >= 14:
        atr[13] = np.nanmean(tr[1:15])  # Initialize with first 14-period average
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Calculate ATR ratio (current ATR / 20-period average ATR) for volatility filter
    atr_ma = np.full(n, np.nan)
    if n >= 34:  # Need 14 + 20 for ATR and its MA
        atr_sum = np.nansum(atr[14:34])  # Sum of first 20 ATR values after warmup
        atr_ma[33] = atr_sum / 20
        for i in range(34, n):
            atr_sum = atr_sum - atr[i-20] + atr[i]
            atr_ma[i] = atr_sum / 20
    
    atr_ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and not np.isnan(atr_ma[i]) and atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(34, n):  # Start after warmup (ATR(14) + ATR MA(20))
        # Skip if any required data is invalid
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(donchian_mid_aligned[i]) or 
            np.isnan(atr_ratio[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses below weekly midpoint
            if close[i] < donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by volatility (inverse volatility scaling)
                vol_scale = min(0.3, max(0.2, 0.25 * (1.2 / atr_ratio[i])))
                signals[i] = vol_scale
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly midpoint
            if close[i] > donchian_mid_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                # Scale position by volatility (inverse volatility scaling)
                vol_scale = min(0.3, max(0.2, 0.25 * (1.2 / atr_ratio[i])))
                signals[i] = -vol_scale
        else:  # Flat
            # Enter long: price breaks above weekly Donchian high with volatility filter
            if (close[i] > donchian_high_aligned[i] and 
                atr_ratio[i] > 1.2):  # Volatility expansion filter
                position = 1
                # Scale position by volatility (inverse volatility scaling)
                vol_scale = min(0.3, max(0.2, 0.25 * (1.2 / atr_ratio[i])))
                signals[i] = vol_scale
            # Enter short: price breaks below weekly Donchian low with volatility filter
            elif (close[i] < donchian_low_aligned[i] and 
                  atr_ratio[i] > 1.2):  # Volatility expansion filter
                position = -1
                # Scale position by volatility (inverse volatility scaling)
                vol_scale = min(0.3, max(0.2, 0.25 * (1.2 / atr_ratio[i])))
                signals[i] = -vol_scale
    
    return signals