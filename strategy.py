#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: Donchian channel breakout on 12h with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high with price above 1d EMA50 and volume > 1.5x average.
# Short when price breaks below 20-period Donchian low with price below 1d EMA50 and volume > 1.5x average.
# Exit when price returns to Donchian midpoint or opposite signal.
# Designed to capture breakouts in trending markets while avoiding false breakouts in ranging conditions.
# Target: 15-25 trades/year to minimize fee decay while capturing high-probability moves.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1d_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 12h
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period average
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(donchian_mid[i]) or np.isnan(ema50_1d_aligned[i]) or \
           np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to midpoint or opposite signal
            if close[i] <= donchian_mid[i] or \
               (close[i] < donchian_low[i] and volume[i] > 1.5 * avg_volume[i] and close[i] < ema50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to midpoint or opposite signal
            if close[i] >= donchian_mid[i] or \
               (close[i] > donchian_high[i] and volume[i] > 1.5 * avg_volume[i] and close[i] > ema50_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: price breaks above Donchian high with volume and above 1d EMA50
            if close[i] > donchian_high[i] and volume_ok and close[i] > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below Donchian low with volume and below 1d EMA50
            elif close[i] < donchian_low[i] and volume_ok and close[i] < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals