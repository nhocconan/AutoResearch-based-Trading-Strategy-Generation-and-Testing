#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Choppiness Index + 1d Supertrend + Volume Spike
# Hypothesis: Choppiness Index identifies ranging markets (CHOP > 61.8) for mean reversion
# and trending markets (CHOP < 38.2) for trend following. Supertrend on 1d filters direction.
# Volume spikes confirm institutional participation. Works in bull via trend following
# and bear via mean reversion in ranges. Target: 20-50 trades/year (80-200 total).

name = "4h_choppiness_supertrend_volume_v1"
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
    
    # Get 1d data for Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Supertrend on 1d (10, 3.0)
    atr_period = 10
    atr_multiplier = 3.0
    
    # Calculate True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0] - low_1d[0], np.abs(high_1d[0] - close_1d[0]), np.abs(low_1d[0] - close_1d[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False).mean().values
    
    # Basic Upper and Lower Bands
    hl_avg = (high_1d + low_1d) / 2
    upper_band = hl_avg + (atr_multiplier * atr)
    lower_band = hl_avg - (atr_multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.zeros_like(close_1d)
    direction = np.ones_like(close_1d)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    for i in range(1, len(close_1d)):
        if close_1d[i-1] > supertrend[i-1]:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
        
        if supertrend[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
    
    # Align Supertrend direction to 4h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1d, direction)
    
    # Choppiness Index on 4h (14-period)
    cp = 14
    # True Range
    tr1_ch = high[1:] - low[1:]
    tr2_ch = np.abs(high[1:] - close[:-1])
    tr3_ch = np.abs(low[1:] - close[:-1])
    tr_ch = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1_ch, np.maximum(tr2_ch, tr3_ch))])
    atr_ch = pd.Series(tr_ch).rolling(window=cp, min_periods=cp).sum().values
    
    # Max and Min high/low over period
    max_hh = pd.Series(high).rolling(window=cp, min_periods=cp).max().values
    min_ll = pd.Series(low).rolling(window=cp, min_periods=cp).min().values
    
    # Chop formula: 100 * log10(sum(atr) / (max(high) - min(low))) / log10(cp)
    chop = np.zeros(n)
    for i in range(cp-1, n):
        if atr_ch[i] > 0 and (max_hh[i] - min_ll[i]) > 0:
            chop[i] = 100 * np.log10(atr_ch[i] / (max_hh[i] - min_ll[i])) / np.log10(cp)
        else:
            chop[i] = 50  # neutral
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if required data not available
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(chop[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: chop indicates strong trend (>61.8) or supertrend turns bearish
            if chop[i] > 61.8 or supertrend_dir_aligned[i] == -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: chop indicates strong trend (>61.8) or supertrend turns bullish
            if chop[i] > 61.8 or supertrend_dir_aligned[i] == 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Chop < 38.2 = trending market, follow supertrend
                if chop[i] < 38.2:
                    if supertrend_dir_aligned[i] == 1:
                        position = 1
                        signals[i] = 0.25
                    elif supertrend_dir_aligned[i] == -1:
                        position = -1
                        signals[i] = -0.25
                # Chop > 61.8 = ranging market, mean reversion at extremes
                elif chop[i] > 61.8:
                    # Mean reversion: buy near support, sell near resistance
                    # Using recent high/low for range boundaries
                    lookback = 20
                    if i >= lookback:
                        recent_high = np.max(high[i-lookback:i+1])
                        recent_low = np.min(low[i-lookback:i+1])
                        range_width = recent_high - recent_low
                        if range_width > 0:
                            # Position in range: 0 = at low, 1 = at high
                            pos_in_range = (close[i] - recent_low) / range_width
                            # Buy in lower 30%, sell in upper 70%
                            if pos_in_range < 0.3:
                                position = 1
                                signals[i] = 0.25
                            elif pos_in_range > 0.7:
                                position = -1
                                signals[i] = -0.25
    
    return signals