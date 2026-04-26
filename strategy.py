#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop_v1
Hypothesis: On 12h timeframe, trade breakouts above/below daily Camarilla R1/S1 only when aligned with 1d EMA50 trend, confirmed by volume spike (>2.0x 20-bar average), and filtered by choppiness regime (CHOP > 50 = range means we mean-revert at extremes, CHOP < 50 = trend means we follow breakouts). Uses ATR(14) stoploss at 2.0x ATR. Discrete sizing at 0.25 to limit fee drag. Target: 12-30 trades/year on BTC/ETH/SOL.
"""

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
    
    # Get 1d data for Camarilla pivot and EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from prior day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use prior day's OHLC (shift by 1 to avoid look-ahead)
    high_prev = np.roll(high_1d, 1)
    low_prev = np.roll(low_1d, 1)
    close_prev = np.roll(close_1d, 1)
    # For first bar, use first available
    high_prev[0] = high_1d[0]
    low_prev[0] = low_1d[0]
    close_prev[0] = close_1d[0]
    
    # Camarilla calculations
    pivot = (high_prev + low_prev + close_prev) / 3.0
    range_val = high_prev - low_prev
    r1 = close_prev + range_val * 1.1 / 12
    s1 = close_prev - range_val * 1.1 / 12
    
    # Calculate 1d EMA50 for trend filter (more stable than EMA34)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align all HTF indicators to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # ATR for stoploss calculation (12h ATR)
    atr_period = 14
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]  # first bar
    atr = pd.Series(tr).ewm(span=atr_period, min_periods=atr_period, adjust=False).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period average (stricter for fewer trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index regime filter (14-period)
    # CHOP > 50 = ranging market (favor mean reversion at extremes)
    # CHOP < 50 = trending market (favor breakouts)
    chop_period = 14
    true_range = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    true_range[0] = high[0] - low[0]
    atr_chop = pd.Series(true_range).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_chop = highest_high - lowest_low
    range_chop = np.where(range_chop == 0, 1e-10, range_chop)
    chop = 100 * np.log10(atr_chop / np.log10(chop_period) / range_chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of pivot calc (1), EMA50 (50), ATR (14), volume MA (20), CHOP (14)
    start_idx = max(1, 50, 14, 20, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        close_val = close[i]
        atr_val = atr[i]
        vol_spike = volume_spike[i]
        chop_val = chop[i]
        
        if position == 0:
            # Determine regime: CHOP < 50 = trending (follow breakouts), CHOP >= 50 = ranging (mean revert at extremes)
            is_trending = chop_val < 50
            
            if is_trending:
                # Trending regime: follow breakouts
                # Long: price breaks above R1, above 1d EMA50, with volume spike
                long_signal = (close_val > r1_val) and (close_val > ema_50_val) and vol_spike
                
                # Short: price breaks below S1, below 1d EMA50, with volume spike
                short_signal = (close_val < s1_val) and (close_val < ema_50_val) and vol_spike
            else:
                # Ranging regime: mean reversion at extremes
                # Long: price breaks below S1 (oversold) and reverts, with volume spike
                long_signal = (close_val < s1_val) and vol_spike
                
                # Short: price breaks above R1 (overbought) and reverts, with volume spike
                short_signal = (close_val > r1_val) and vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit conditions
            if is_trending:
                # In trending regime: exit on break below S1 OR ATR stop
                exit_signal = (close_val < s1_val) or (close_val < entry_price - 2.0 * atr_val)
            else:
                # In ranging regime: exit on reversion to pivot OR ATR stop
                exit_signal = (close_val >= pivot) or (close_val < entry_price - 2.0 * atr_val)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit conditions
            if is_trending:
                # In trending regime: exit on break above R1 OR ATR stop
                exit_signal = (close_val > r1_val) or (close_val > entry_price + 2.0 * atr_val)
            else:
                # In ranging regime: exit on reversion to pivot OR ATR stop
                exit_signal = (close_val <= pivot) or (close_val > entry_price + 2.0 * atr_val)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeChop_v1"
timeframe = "12h"
leverage = 1.0