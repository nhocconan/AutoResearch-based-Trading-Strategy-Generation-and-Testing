#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour timeframe with 1-day Choppiness Index regime filter and 1-day ATR-based breakout.
# The Choppiness Index identifies ranging (CHOP > 61.8) vs trending (CHOP < 38.2) markets.
# In trending markets, we trade breakouts of the 1-day ATR-based channel (similar to Donchian but volatility-adjusted).
# In ranging markets, we fade moves toward the 1-day VWAP with reversion to the mean.
# This adaptive approach aims to reduce whipsaws in chop while capturing trends, suitable for 2021-2026 BTC/ETH markets.
# Position size: 0.25. Target trades: 20-50 per year per symbol (80-200 total over 4 years).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1-day data ONCE for regime and signals
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 30:  # Need enough for CHOP and ATR
        return np.zeros(n)
    
    # Calculate 1-day Choppiness Index (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = np.zeros(len(df_1d))
    tr = np.maximum(high_1d - low_1d, np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]  # First TR
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero
    atr_safe = np.where(atr_1d == 0, 1e-10, atr_1d)
    sum_tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_tr_14 / (atr_safe * 14)) / np.log10(14)
    
    # Calculate 1-day ATR-based channels (like Donchian but ATR-based)
    atr_mult = 1.0
    upper_channel = pd.Series(close_1d).rolling(window=20, min_periods=20).max().values + atr_mult * atr_1d
    lower_channel = pd.Series(close_1d).rolling(window=20, min_periods=20).min().values - atr_mult * atr_1d
    
    # Calculate 1-day VWAP for mean reversion in ranging markets
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap = (pd.Series(typical_price * volume).rolling(window=20, min_periods=20).sum().values / 
            pd.Series(volume).rolling(window=20, min_periods=20).sum().values.replace(0, 1e-10))
    
    # Align all 1D indicators to 12H timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    upper_channel_aligned = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_aligned = align_htf_to_ltf(prices, df_1d, lower_channel)
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(chop_aligned[i]) or 
            np.isnan(upper_channel_aligned[i]) or
            np.isnan(lower_channel_aligned[i]) or
            np.isnan(vwap_aligned[i])):
            signals[i] = 0.0
            continue
        
        chop_val = chop_aligned[i]
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            if chop_val < 38.2:  # Trending market
                # Breakout long: price above upper channel
                if close[i] > upper_channel_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Breakout short: price below lower channel
                elif close[i] < lower_channel_aligned[i]:
                    position = -1
                    signals[i] = -position_size
            elif chop_val > 61.8:  # Ranging market
                # Mean reversion long: price near lower channel and below VWAP
                if close[i] < lower_channel_aligned[i] * 1.02 and close[i] < vwap_aligned[i]:
                    position = 1
                    signals[i] = position_size
                # Mean reversion short: price near upper channel and above VWAP
                elif close[i] > upper_channel_aligned[i] * 0.98 and close[i] > vwap_aligned[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: based on regime
            if chop_val < 38.2:  # Trending: exit on reversal below lower channel
                if close[i] < lower_channel_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:  # Ranging: exit on mean reversion to VWAP
                if close[i] >= vwap_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
        elif position == -1:
            # Exit short: based on regime
            if chop_val < 38.2:  # Trending: exit on reversal above upper channel
                if close[i] > upper_channel_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:  # Ranging: exit on mean reversion to VWAP
                if close[i] <= vwap_aligned[i]:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
    
    return signals

name = "12h_1d_CHOP_ATR_VWAP_Adaptive_v1"
timeframe = "12h"
leverage = 1.0