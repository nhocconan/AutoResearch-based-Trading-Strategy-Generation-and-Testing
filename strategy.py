#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Choppiness Index regime filter + 1d Donchian breakout with volume confirmation
# Uses weekly trend filter to avoid counter-trend trades. Choppiness > 61.8 = range (mean revert),
# Choppiness < 38.2 = trending (breakout follow). Designed for low-frequency trades
# (<150 total) to minimize fee drag and work in both bull/bear markets by adapting to regime.

name = "12h_Chop_Donchian_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian breakout levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate daily Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # 20-period rolling max/min (use previous day's values to avoid look-ahead)
    # Using pandas rolling with min_periods
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Calculate 1w EMA40 for trend filter
    close_1w = df_1w['close'].values
    ema40_1w = pd.Series(close_1w).ewm(span=40, adjust=False, min_periods=40).mean().values
    ema40_1w_aligned = align_htf_to_ltf(prices, df_1w, ema40_1w)
    
    # Calculate Choppiness Index on 12h data (14-period)
    # CHOP = 100 * log10(sum(ATR) / (max(high) - min(low))) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = 0  # First TR is just high-low
    
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    range_hl = max_high - min_low
    
    # Avoid division by zero
    chop = np.full_like(close, 50.0)  # Default to neutral
    mask = (range_hl > 0) & (~np.isnan(atr_sum))
    chop[mask] = 100 * np.log10(atr_sum[mask] / range_hl[mask]) / np.log10(14)
    
    # Volume spike (1.5x 20-period EMA)
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure all indicators have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(upper_20_aligned[i]) or np.isnan(lower_20_aligned[i]) or 
            np.isnan(ema40_1w_aligned[i]) or np.isnan(chop[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime: Choppiness < 38.2 = trending, > 61.8 = ranging
            is_trending = chop[i] < 38.2
            is_ranging = chop[i] > 61.8
            
            if is_trending:
                # In trending regime: follow Donchian breakouts with weekly trend filter
                # Enter long: price breaks above upper Donchian with weekly uptrend and volume spike
                if (close[i] > upper_20_aligned[i] and 
                    close[i] > ema40_1w_aligned[i] and vol_spike[i]):
                    signals[i] = 0.25
                    position = 1
                # Enter short: price breaks below lower Donchian with weekly downtrend and volume spike
                elif (close[i] < lower_20_aligned[i] and 
                      close[i] < ema40_1w_aligned[i] and vol_spike[i]):
                    signals[i] = -0.25
                    position = -1
            elif is_ranging:
                # In ranging regime: mean reversion at Donchian boundaries
                # Enter long: price touches lower Donchian with weekly uptrend bias and volume spike
                if (close[i] <= lower_20_aligned[i] * 1.001 and  # Allow small buffer
                    close[i] > ema40_1w_aligned[i] and vol_spike[i]):
                    signals[i] = 0.20
                    position = 1
                # Enter short: price touches upper Donchian with weekly downtrend bias and volume spike
                elif (close[i] >= upper_20_aligned[i] * 0.999 and  # Allow small buffer
                      close[i] < ema40_1w_aligned[i] and vol_spike[i]):
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Exit long: price crosses midline or weekly trend fails
            midline = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if (close[i] < midline or 
                close[i] < ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses midline or weekly trend fails
            midline = (upper_20_aligned[i] + lower_20_aligned[i]) / 2
            if (close[i] > midline or 
                close[i] > ema40_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals