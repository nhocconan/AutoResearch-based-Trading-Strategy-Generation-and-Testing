#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d EMA200 trend filter.
# Long when: Price breaks above 4h Donchian(20) high + volume > 1.5x avg + price > 1d EMA200
# Short when: Price breaks below 4h Donchian(20) low + volume > 1.5x avg + price < 1d EMA200
# Exit when: Price crosses opposite Donchian band (long exit at low, short exit at high)
# Uses 4h for structure/entry, 1d for trend filter, volume for confirmation.
# Target: 15-30 trades/year per symbol (~60-120 total over 4 years).
# Session filter: 08-20 UTC to avoid low-liquidity hours.
name = "4h_Donchian_Volume_EMA200_Trend"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    session_mask = (hours >= 8) & (hours <= 20)
    
    # Get 4h Donchian channels (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    donch_high_4h = pd.Series(df_4h['high'].values).rolling(window=20, min_periods=20).max().values
    donch_low_4h = pd.Series(df_4h['low'].values).rolling(window=20, min_periods=20).min().values
    donch_high_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_high_4h)
    donch_low_4h_aligned = align_htf_to_ltf(prices, df_4h, donch_low_4h)
    
    # Get 1d EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after all indicators are ready
    start_idx = max(200, 20)  # EMA200 needs 200, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not session_mask[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is not available
        if (np.isnan(donch_high_4h_aligned[i]) or np.isnan(donch_low_4h_aligned[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donch_high = donch_high_4h_aligned[i]
        donch_low = donch_low_4h_aligned[i]
        ema200 = ema200_1d_aligned[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        if position == 0:
            # Long entry: Price breaks above Donchian high + volume spike + above 1d EMA200
            if price > donch_high and vol > 1.5 * vol_ma and price > ema200:
                signals[i] = 0.20
                position = 1
            # Short entry: Price breaks below Donchian low + volume spike + below 1d EMA200
            elif price < donch_low and vol > 1.5 * vol_ma and price < ema200:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below Donchian low
            if price < donch_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: Price crosses above Donchian high
            if price > donch_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals