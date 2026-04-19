#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d trend filter and volume confirmation
# Uses Donchian channel (20-period high/low) to identify breakouts
# Filters by 1-day EMA50 trend and volume spike to avoid false breakouts
# Exits when price re-enters Donchian channel or reverses
# Target: 20-50 trades/year to avoid fee drag
name = "4h_DonchianBreakout_Trend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for multi-timeframe analysis (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h Donchian channel (20-period high/low)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h ATR for position sizing and stops
    tr = np.maximum(high - low, np.absolute(high - np.roll(close, 1)), np.absolute(low - np.roll(close, 1)))
    tr[0] = high[0] - low[0]
    atr_4h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_4h[i]
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        if position == 0:
            # Long: Price breaks above Donchian high + volume + 1d uptrend
            if price > donch_high[i] and volume_filter and price > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + volume + 1d downtrend
            elif price < donch_low[i] and volume_filter and price < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Price re-enters Donchian channel or 1d trend reverses
            if price < donch_high[i] or price < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Price re-enters Donchian channel or 1d trend reverses
            if price > donch_low[i] or price > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals