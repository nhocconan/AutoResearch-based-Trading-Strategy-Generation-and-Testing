#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction (R4/S4 breakout continuation) and volume confirmation
# - Long when price breaks above 6h Donchian(20) high AND price > 1d weekly pivot R4 with volume spike
# - Short when price breaks below 6h Donchian(20) low AND price < 1d weekly pivot S4 with volume spike
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss: exit when price moves against position by 2.0x ATR(6h,14) or price reverts to 6h Donchian midpoint
# - Targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Weekly pivot (from 1d) provides structural bias: R4/S4 are extreme levels where breakouts have higher follow-through

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Pre-compute 1d indicators for weekly pivot (using prior week's daily data)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate weekly pivot points from prior week's daily OHLC
    # Need to group daily data into weeks (starting Monday)
    # For simplicity, use rolling window of 5 days (approx 1 week) to get weekly high/low/close
    # Weekly high = max(high over prior 5 days)
    # Weekly low = min(low over prior 5 days)
    # Weekly close = close of prior day (yesterday's close)
    if len(high_1d) >= 5:
        weekly_high = pd.Series(high_1d).rolling(window=5, min_periods=5).max().shift(1).values  # prior week
        weekly_low = pd.Series(low_1d).rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = pd.Series(close_1d).shift(1).values  # yesterday's close
    else:
        weekly_high = np.full_like(high_1d, np.nan)
        weekly_low = np.full_like(low_1d, np.nan)
        weekly_close = np.full_like(close_1d, np.nan)
    
    # Weekly pivot calculation (standard formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    # Weekly Camarilla-like levels (R4/S4 are extreme breakout levels)
    r4 = weekly_pivot + (weekly_range * 1.1 / 2)  # R4 = pivot + 1.1*range/2
    s4 = weekly_pivot - (weekly_range * 1.1 / 2)  # S4 = pivot - 1.1*range/2
    
    # Align weekly pivot levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 1d volume confirmation: > 1.8x 20-period average (stricter for fewer trades)
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = volume_1d > (1.8 * avg_volume_20_1d)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # 6h Donchian(20) breakout levels
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().shift(1).values  # prior 20 periods
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().shift(1).values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 6h ATR(14) for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1)) if 'close_6h' in locals() else np.abs(high_6h - np.roll(prices['close'].values, 1))
    tr3 = np.abs(low_6h - np.roll(prices['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14_6h = np.zeros_like(tr)
    atr_14_6h[14-1] = np.mean(tr[:14])
    for i in range(14, len(tr)):
        atr_14_6h[i] = (atr_14_6h[i-1] * (14-1) + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    entry_atr = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_spike_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(atr_14_6h[i])):
            signals[i] = 0.0
            continue
        
        close_price = prices['close'].iloc[i]
        
        if position == 1:  # Long position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint (mean reversion)
            if (close_price < entry_price - 2.0 * entry_atr or 
                close_price > donchian_high[i]):  # Take profit at breakout level
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: ATR-based stoploss or price reverts to Donchian midpoint
            if (close_price > entry_price + 2.0 * entry_atr or 
                close_price < donchian_low[i]):  # Take profit at breakout level
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with weekly pivot alignment and volume spike
            if vol_spike_1d_aligned[i]:
                # Long signal: price breaks above Donchian(20) high AND above weekly R4
                if (close_price > donchian_high[i] and 
                    close_price > r4_aligned[i]):
                    position = 1
                    entry_price = close_price
                    entry_atr = atr_14_6h[i]
                    signals[i] = 0.25
                # Short signal: price breaks below Donchian(20) low AND below weekly S4
                elif (close_price < donchian_low[i] and 
                      close_price < s4_aligned[i]):
                    position = -1
                    entry_price = close_price
                    entry_atr = atr_14_6h[i]
                    signals[i] = -0.25
    
    return signals