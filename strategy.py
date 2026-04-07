#!/usr/bin/env python3
"""
1d_1w_donchian_breakout_volume_filter_v1
Hypothesis: On daily timeframe, use Donchian(20) breakout with weekly trend filter and volume confirmation to capture strong trends while minimizing whipsaws. The 1d timeframe balances responsiveness with reduced noise, and volume filters ensure institutional participation. Designed for 30-100 total trades over 4 years (~7-25/year) to minimize fee drift while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian_breakout_volume_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 20-period average on 1d timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend filter: EMA(21) on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price breaks above Donchian high with weekly uptrend
                if close[i] > donchian_high[i] and close[i] > ema_21_1w_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish breakout: price breaks below Donchian low with weekly downtrend
                elif close[i] < donchian_low[i] and close[i] < ema_21_1w_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals