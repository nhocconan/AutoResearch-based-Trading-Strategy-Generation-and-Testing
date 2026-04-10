#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h trend filter (HMA21) and volume confirmation
# - Long when price breaks above Donchian(20) high AND 12h HMA21 is rising AND volume > 1.5x 20-bar avg
# - Short when price breaks below Donchian(20) low AND 12h HMA21 is falling AND volume > 1.5x 20-bar avg
# - Exit on opposite Donchian breakout or when volume drops below average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets ~25 trades/year (100 total over 4 years) to avoid fee drag
# - 12h HMA trend filter reduces false signals in choppy markets
# - Volume confirmation ensures institutional participation
# - Donchian channels provide clear structure in both bull and bear markets

name = "4h_12h_donchian_breakout_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h indicators
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # 12h HMA(21) for trend filter
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    def hma(values, window):
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        wma_half = np.array([wma(values[i:i+half_window], half_window) 
                            for i in range(len(values) - half_window + 1)])
        wma_full = np.array([wma(values[i:i+window], window) 
                            for i in range(len(values) - window + 1)])
        hma_raw = 2 * wma_half - wma_full[:len(wma_half)]
        return np.array([wma(hma_raw[i:i+sqrt_window], sqrt_window) 
                        for i in range(len(hma_raw) - sqrt_window + 1)])
    
    # Calculate HMA with proper padding for alignment
    hma_21_12h = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 21:
        hma_values = hma(close_12h, 21)
        start_idx = 21 - 1  # HMA(21) needs 21 periods
        end_idx = start_idx + len(hma_values)
        hma_21_12h[start_idx:end_idx] = hma_values
    
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # 12h HMA slope (rising/falling)
    hma_slope = np.diff(hma_21_12h_aligned, prepend=hma_21_12h_aligned[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    
    # 12h volume confirmation: > 1.5x 20-period average
    volume_12h = df_12h['volume'].values
    avg_volume_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_spike_12h = volume_12h > (1.5 * avg_volume_20_12h)
    vol_spike_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_spike_12h)
    
    # Pre-compute 4h Donchian channels
    highest_high_20 = prices['high'].rolling(window=20, min_periods=20).max().values
    lowest_low_20 = prices['low'].rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_21_12h_aligned[i]) or np.isnan(vol_spike_12h_aligned[i]) or 
            np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long signal: Donchian breakout above + 12h HMA rising + volume spike
            if (prices['close'].iloc[i] > highest_high_20[i] and 
                hma_rising[i] and 
                vol_spike_12h_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short signal: Donchian breakdown below + 12h HMA falling + volume spike
            elif (prices['close'].iloc[i] < lowest_low_20[i] and 
                  hma_falling[i] and 
                  vol_spike_12h_aligned[i]):
                position = -1
                signals[i] = -0.25
        else:  # Have position - look for exit
            # Exit long on Donchian breakdown or loss of volume/Trend
            if position == 1 and (prices['close'].iloc[i] < lowest_low_20[i] or 
                                 not hma_rising[i] or 
                                 not vol_spike_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Exit short on Donchian breakout above or loss of volume/Trend
            elif position == -1 and (prices['close'].iloc[i] > highest_high_20[i] or 
                                    not hma_falling[i] or 
                                    not vol_spike_12h_aligned[i]):
                position = 0
                signals[i] = 0.0
            # Hold position otherwise
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals