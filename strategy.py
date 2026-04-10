#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# - Donchian(20) breakout provides clear entry/exit levels with proven edge on SOL/ETH
# - 12h HMA(21) trend filter ensures trades align with higher timeframe momentum
# - Volume confirmation: current 4h volume > 1.5x 20-period average filters weak breakouts
# - Designed for 4h timeframe: targets 20-50 trades/year (80-200 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: breakouts capture momentum, HMA filter avoids counter-trend trades
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "4h_12h_donchian_hma_volume_v1"
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
    
    # Pre-compute 12h HMA(21)
    close_12h = df_12h['close'].values
    # HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(close_12h, half_len)
    wma_full = wma(close_12h, 21)
    wma_2xhalf_minus_full = 2 * wma_half - wma_full
    hma_12h = wma(wma_2xhalf_minus_full, sqrt_len)
    
    # Pad HMA to match original length
    hma_12h_padded = np.full_like(close_12h, np.nan)
    hma_12h_padded[half_len-1:half_len-1+len(hma_12h)] = hma_12h
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    
    # Pre-compute 4h Donchian channels (20-period)
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Pre-compute 4h volume confirmation
    volume_4h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_4h > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(hma_12h_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (trend reversal)
            if prices['close'].iloc[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (trend reversal)
            if prices['close'].iloc[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakout with trend and volume filters
            if vol_spike[i]:
                # Breakout long: price closes above Donchian high AND above 12h HMA (bullish alignment)
                if prices['close'].iloc[i] > donchian_high[i] and prices['close'].iloc[i] > hma_12h_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price closes below Donchian low AND below 12h HMA (bearish alignment)
                elif prices['close'].iloc[i] < donchian_low[i] and prices['close'].iloc[i] < hma_12h_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals