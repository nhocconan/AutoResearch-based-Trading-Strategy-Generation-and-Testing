#!/usr/bin/env python3
# 4h_donchian_breakout_1d_volume_v1
# Hypothesis: 4-hour Donchian channel breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with volume > 1.5x average and 1-day EMA(50) up.
# Short when price breaks below 20-period Donchian low with volume > 1.5x average and 1-day EMA(50) down.
# Exit when price returns to Donchian midpoint or opposite signal.
# Designed to capture breakouts in both bull and bear markets with trend alignment.
# Target: 25-35 trades/year to minimize fee drag while capturing high-probability breakouts.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_breakout_1d_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(donchian_mid[i]) or np.isnan(avg_volume[i]) or \
           np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to Donchian midpoint or opposite signal
            if close[i] <= donchian_mid[i] or \
               (close[i] < donchian_low[i] and volume[i] > 1.5 * avg_volume[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to Donchian midpoint or opposite signal
            if close[i] >= donchian_mid[i] or \
               (close[i] > donchian_high[i] and volume[i] > 1.5 * avg_volume[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            # Trend filter: price above/below 1-day EMA50
            price_vs_ema = close[i] > ema50_1d_aligned[i]
            
            # Long entry: price breaks above Donchian high with volume and trend alignment
            if close[i] > donchian_high[i] and volume_ok and price_vs_ema:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and trend alignment
            elif close[i] < donchian_low[i] and volume_ok and not price_vs_ema:
                position = -1
                signals[i] = -0.25
    
    return signals