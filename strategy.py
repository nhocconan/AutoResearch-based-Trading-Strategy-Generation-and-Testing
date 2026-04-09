#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout (20) + 1d ATR filter + volume confirmation
# Donchian breakouts capture strong momentum moves; ATR filter ensures sufficient volatility
# Volume confirmation (2.0x 20-period avg) filters weak breakouts
# Works in bull/bear: breakouts work in both directions with volume/ATR confirmation
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_donchian_atr_volume_v1"
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
    
    # Load 1d data ONCE before loop for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan  # First value has no previous close
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            highest_high[i] = np.max(high[i-20:i])
            lowest_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_14_1d_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0x 20-period average
        volume_confirmed = volume[i] > 2.0 * avg_volume[i]
        # ATR filter: ensure sufficient volatility (ATR > 0.5% of price)
        atr_filter = atr_14_1d_aligned[i] > 0.005 * close[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian lower band OR ATR filter fails
            if close[i] < lowest_low[i] or not atr_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian upper band OR ATR filter fails
            if close[i] > highest_high[i] or not atr_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation, ATR filter, and Donchian breakout
            if volume_confirmed and atr_filter:
                # Long entry: price > Donchian upper band (bullish breakout)
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian lower band (bearish breakout)
                elif close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals