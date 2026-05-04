#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h HMA(21) trend + volume confirmation
# In trending markets (12h HMA rising), we trade breakouts in trend direction: long on upper breakout, short on lower breakout.
# In ranging markets (12h HMA flat), we fade extremes: short near upper band, long near lower band.
# Volume confirmation (>1.3x 20-period EMA) reduces false breakouts. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.

name = "4h_Donchian20_12hHMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for HMA trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h HMA(21)
    close_12h = pd.Series(df_12h['close'])
    wma_half = close_12h.rolling(window=10, min_periods=10).apply(
        lambda x: np.average(x, weights=np.arange(1, 11)), raw=True
    )
    wma_full = close_12h.rolling(window=21, min_periods=21).apply(
        lambda x: np.average(x, weights=np.arange(1, 22)), raw=True
    )
    hma_12h = 2 * wma_half - wma_full
    hma_12h = hma_12h.rolling(window=5, min_periods=5).apply(
        lambda x: np.average(x, weights=np.arange(1, 6)), raw=True
    ).values
    
    # Calculate 12h HMA slope (rising/falling/flat)
    hma_slope = np.diff(hma_12h, prepend=hma_12h[0])
    hma_rising = hma_slope > 0
    hma_falling = hma_slope < 0
    hma_flat = np.abs(hma_slope) <= 1e-7  # essentially flat
    
    # Align 12h HMA and slope to 4h timeframe
    hma_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    hma_rising_aligned = align_htf_to_ltf(prices, df_12h, hma_rising.astype(float))
    hma_falling_aligned = align_htf_to_ltf(prices, df_12h, hma_falling.astype(float))
    hma_flat_aligned = align_htf_to_ltf(prices, df_12h, hma_flat.astype(float))
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_upper = highest_high.values
    donchian_lower = lowest_low.values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume confirmation: 20-period EMA of volume on 4h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(hma_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.3 x 20-period EMA
        volume_confirm = volume[i] > (1.3 * vol_ema_20[i])
        
        if position == 0:
            # Determine trend: rising (HMA up), falling (HMA down), or flat (HMA flat)
            if hma_rising_aligned[i] > 0.5:
                # Uptrend: long on upper breakout
                if close[i] > donchian_upper[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
            elif hma_falling_aligned[i] > 0.5:
                # Downtrend: short on lower breakout
                if close[i] < donchian_lower[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
            else:
                # Ranging market (HMA flat): fade extremes
                if close[i] <= donchian_lower[i] and volume_confirm:
                    signals[i] = 0.25
                    position = 1
                elif close[i] >= donchian_upper[i] and volume_confirm:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long: price retouches midpoint OR trend changes to downtrend
            if (close[i] <= donchian_mid[i] or 
                hma_falling_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retouches midpoint OR trend changes to uptrend
            if (close[i] >= donchian_mid[i] or 
                hma_rising_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals