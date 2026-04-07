#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 4h Donchian(20) Breakout + 1d Trend + Volume Spike
# Hypothesis: Donchian breakouts capture strong directional moves. 1d trend filter ensures
# alignment with higher-timeframe momentum. Volume spikes confirm institutional participation.
# Works in bull markets via breakouts above upper band + uptrend, in bear via breakouts
# below lower band + downtrend. Designed for low trade frequency to minimize fee drag.
# Target: 19-50 trades/year (75-200 total over 4 years) for 4h timeframe.

name = "4h_donchian20_1d_trend_volume_v1"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Check volume confirmation
        vol_ok = vol_spike[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower or trend turns bearish
            if close[i] < low_20[i] or close[i] < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper or trend turns bullish
            if close[i] > high_20[i] or close[i] > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Breakout above upper band + uptrend
                if close[i] > high_20[i] and close[i] > ema_20_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout below lower band + downtrend
                elif close[i] < low_20[i] and close[i] < ema_20_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals