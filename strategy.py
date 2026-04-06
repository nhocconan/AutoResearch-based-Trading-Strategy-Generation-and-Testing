#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(20) trend + volume confirmation with volume spike
# Enter long when: price breaks above Donchian high(20), price > EMA(20) on 1w, volume > 1.5x average
# Enter short when: price breaks below Donchian low(20), price < EMA(20) on 1w, volume > 1.5x average
# Exit when: opposite Donchian break occurs or volume drops below average
# Uses higher timeframe trend filter to avoid counter-trend trades
# Target: 30-100 trades over 4 years by combining trend filter with breakout logic

name = "1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
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
    
    # Donchian channels (20-period) on 1d
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below Donchian low OR volume drops below average
            if low[i] <= donchian_low[i] or volume[i] < pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high OR volume drops below average
            if high[i] >= donchian_high[i] or volume[i] < pd.Series(volume).rolling(window=20, min_periods=20).mean().values[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian break + EMA trend + volume spike
            if volume[i] > volume_threshold[i]:
                if high[i] > donchian_high[i] and close[i] > ema_20_1w_aligned[i]:
                    # Bullish breakout with uptrend on 1w
                    signals[i] = 0.25
                    position = 1
                elif low[i] < donchian_low[i] and close[i] < ema_20_1w_aligned[i]:
                    # Bearish breakout with downtrend on 1w
                    signals[i] = -0.25
                    position = -1
    
    return signals