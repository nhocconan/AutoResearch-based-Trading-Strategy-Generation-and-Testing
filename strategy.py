#!/usr/bin/env python3
"""
6h_keltner_breakout_1d_trend_volume_v1
Hypothesis: On 6-hour timeframe, use Keltner Channel breakout with 1-day trend filter and volume confirmation to capture trend continuations while avoiding false breakouts. The 6h timeframe reduces noise, Keltner Channels adapt to volatility, and 1d trend ensures alignment with higher timeframe momentum. Volume filters ensure institutional participation. Designed for 50-150 total trades over 4 years (~12-37/year) to minimize fee drag while performing in both bull and bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

name = "6h_keltner_breakout_1d_trend_volume_v1"
timeframe = "6h"
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
    
    # Keltner Channel parameters
    kc_period = 20
    kc_mult = 2.0
    
    # Calculate Keltner Channel
    close_series = pd.Series(close)
    middle = close_series.ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=kc_period, adjust=False, min_periods=kc_period).mean().values
    
    upper = middle + kc_mult * atr
    lower = middle - kc_mult * atr
    
    # Volume filter: 20-period average on 6h timeframe
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # 1d trend filter (Higher Timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(kc_period, 20), n):
        # Skip if data not available
        if (np.isnan(middle[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation: require volume above average
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below middle line
            if close[i] < middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above middle line
            if close[i] > middle[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if vol_ok:
                # Bullish breakout: price closes above upper line AND 1d trend up
                if close[i] > upper[i] and close[i-1] <= upper[i-1] and close[i] > ema_50_1d_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Bearish breakout: price closes below lower line AND 1d trend down
                elif close[i] < lower[i] and close[i-1] >= lower[i-1] and close[i] < ema_50_1d_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals