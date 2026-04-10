#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot structure + volume confirmation
# - Long: price > Donchian high(20) AND price > 1d weekly pivot R1 AND volume > 1.5x 20-period average
# - Short: price < Donchian low(20) AND price < 1d weekly pivot S1 AND volume > 1.5x 20-period average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - ATR-based stoploss (2.0x ATR(14)) to manage risk
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Weekly pivot levels from 1d data provide structural support/resistance that works in both bull and bear markets

name = "6h_1d_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d weekly pivot points (using prior week's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    open_1d = df_1d['open'].values
    
    # Calculate weekly OHLC from daily data (group by week)
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().values
    weekly_open = pd.Series(open_1d).rolling(window=7, min_periods=7).first().values
    
    # Weekly pivot calculation: P = (H + L + C) / 3
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    # Weekly R1 = 2*P - L
    weekly_r1 = 2 * weekly_pivot - weekly_low
    # Weekly S1 = 2*P - H
    weekly_s1 = 2 * weekly_pivot - weekly_high
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # Pre-compute 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (1.5 * avg_volume_20)
    
    # Pre-compute 6h ATR(14) for stoploss
    tr1_6h = high_6h - low_6h
    tr2_6h = np.abs(high_6h - np.roll(close_6h, 1))
    tr3_6h = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr1_6h, np.maximum(tr2_6h, tr3_6h))
    tr_6h[0] = tr1_6h[0]
    atr_14 = pd.Series(tr_6h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_r1_aligned[i]) or np.isnan(weekly_s1_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR stoploss hit
            if close_6h[i] < donchian_low[i] or close_6h[i] < entry_price - 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR stoploss hit
            if close_6h[i] > donchian_high[i] or close_6h[i] > entry_price + 2.0 * atr_14[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with weekly pivot filter and volume confirmation
            if vol_spike[i]:
                # Long: price > Donchian high(20) AND price > weekly R1
                if close_6h[i] > donchian_high[i] and close_6h[i] > weekly_r1_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price < Donchian low(20) AND price < weekly S1
                elif close_6h[i] < donchian_low[i] and close_6h[i] < weekly_s1_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals