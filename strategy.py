#!/usr/bin/env python3
"""
6h_Bollinger_Bands_Squeeze_Breakout
Breakout strategy that trades Bollinger Band squeezes on 6h timeframe with 1d trend filter.
Long when BB width < 20th percentile (squeeze) and price breaks above upper band with volume confirmation.
Short when BB width < 20th percentile (squeeze) and price breaks below lower band with volume confirmation.
Exit when price returns to middle band or BB width expands above 50th percentile.
Uses 1d EMA50 to filter direction: only long in uptrend, only short in downtrend.
Designed to work in both bull (breakouts continue) and bear (mean reversion after squeeze) markets.
Target: 15-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands parameters
    bb_period = 20
    bb_std = 2.0
    
    # Calculate Bollinger Bands
    sma = np.full(n, np.nan)
    std_dev = np.full(n, np.nan)
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    bb_width = np.full(n, np.nan)
    
    for i in range(bb_period - 1, n):
        sma[i] = np.mean(close[i - bb_period + 1:i + 1])
        std_dev[i] = np.std(close[i - bb_period + 1:i + 1])
        upper_band[i] = sma[i] + bb_std * std_dev[i]
        lower_band[i] = sma[i] - bb_std * std_dev[i]
        middle_band[i] = sma[i]
        bb_width[i] = (upper_band[i] - lower_band[i]) / sma[i] * 100  # percentage
    
    # Calculate percentiles of BB width for squeeze detection
    bb_width_pct = np.full(n, np.nan)
    lookback = 50
    for i in range(lookback - 1, n):
        window = bb_width[i - lookback + 1:i + 1]
        valid_window = window[~np.isnan(window)]
        if len(valid_window) > 0:
            percentile_20 = np.percentile(valid_window, 20)
            percentile_50 = np.percentile(valid_window, 50)
            bb_width_pct[i] = (bb_width[i] - percentile_20) / (percentile_50 - percentile_20 + 1e-10) * 100
            bb_width_pct[i] = np.clip(bb_width_pct[i], 0, 100)
    
    # Volume ratio (current vs 20-period average)
    vol_ma = np.full(n, np.nan)
    vol_ratio = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 6h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need BB, volume, and EMA1d
    start_idx = max(bb_period - 1, lookback - 1, 19, ema_1d_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(bb_width_pct[i]) or np.isnan(vol_ratio[i]) or 
            np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(middle_band[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        bb_width_percentile = bb_width_pct[i]
        vol_ratio_val = vol_ratio[i]
        upper = upper_band[i]
        lower = lower_band[i]
        middle = middle_band[i]
        ema1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Look for squeeze breakout with volume confirmation
            is_squeeze = bb_width_percentile < 20  # BB width in lower 20th percentile
            has_volume = vol_ratio_val > 1.5       # 50% above average volume
            
            if is_squeeze and has_volume:
                # Long: break above upper band in uptrend
                if price > upper and price > ema1d_val:
                    signals[i] = size
                    position = 1
                # Short: break below lower band in downtrend
                elif price < lower and price < ema1d_val:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: return to middle band or BB expansion (squeeze ended)
            if price < middle or bb_width_percentile > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: return to middle band or BB expansion (squeeze ended)
            if price > middle or bb_width_percentile > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Bollinger_Bands_Squeeze_Breakout"
timeframe = "6h"
leverage = 1.0