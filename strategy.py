#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w pivot direction and 1d volume confirmation
# - Primary signal: Price breaks above/below Donchian(20) channel on 6h
# - HTF filter: Trade only in direction of weekly Camarilla pivot bias (above/below weekly pivot)
# - Volume filter: 1d volume > 1.3x 20-period average volume (institutional participation)
# - Position size: 0.25 discrete level to minimize fee churn
# - Stoploss: 2.0x ATR(20) on 6h
# - Target: 12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Weekly pivot provides structural bias that works in both bull/bear markets
# - Volume confirmation ensures breakouts have conviction
# - Discrete sizing and filters reduce overtrading and fee drag

name = "6h_1w_1d_donchian_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 20 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute weekly Camarilla pivot for directional bias
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot point (Camarilla)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    # Weekly resistance/support levels (Camarilla)
    range_1w = high_1w - low_1w
    r3_1w = pivot_1w + range_1w * 1.1 / 2  # R3 level
    s3_1w = pivot_1w - range_1w * 1.1 / 2  # S3 level
    
    # Determine weekly bias: 1 = bullish (above pivot), -1 = bearish (below pivot), 0 = neutral
    weekly_bias = np.zeros(len(close_1w))
    weekly_bias[close_1w > pivot_1w] = 1
    weekly_bias[close_1w < pivot_1w] = -1
    weekly_bias_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias)
    
    # Pre-compute 1d volume spike filter
    volume_1d = df_1d['volume'].values
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (1.3 * avg_volume_20)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike)
    
    # Pre-compute 6h Donchian channels (20-period)
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 6h ATR(20) for stoploss
    tr_6h1 = high_6h - low_6h
    tr_6h2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr_6h3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr_6h = np.maximum(tr_6h1, np.maximum(tr_6h2, tr_6h3))
    tr_6h[0] = tr_6h1[0]
    atr_20 = pd.Series(tr_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_spike_aligned[i]) or np.isnan(weekly_bias_aligned[i]) or
            np.isnan(atr_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] < donchian_low[i] or close_6h[i] < entry_price - 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Donchian mean reversion OR stoploss hit
            if close_6h[i] > donchian_high[i] or close_6h[i] > entry_price + 2.0 * atr_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with weekly bias and volume confirmation
            if vol_spike_aligned[i]:
                bias = weekly_bias_aligned[i]
                # Long: price breaks above Donchian high with bullish weekly bias
                if close_6h[i] > donchian_high[i] and bias == 1:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below Donchian low with bearish weekly bias
                elif close_6h[i] < donchian_low[i] and bias == -1:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals