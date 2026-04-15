#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR-based volatility filter
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate 14-day ATR for volatility filter
    tr1 = np.abs(daily_high[1:] - daily_low[1:])
    tr2 = np.abs(daily_high[1:] - daily_close[:-1])
    tr3 = np.abs(daily_low[1:] - daily_close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 4-hour Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4-hour EMA(50) for trend filter
    ema50 = pd.Series(close).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align daily ATR to 4h timeframe (wait for daily close)
    atr_aligned = align_htf_to_ltf(prices, daily, atr)
    
    signals = np.zeros(n)
    
    for i in range(200, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade when ATR is above its 50-period median
        atr_median = pd.Series(atr_aligned[:i+1]).rolling(window=50, min_periods=50).median().iloc[-1]
        vol_filter = atr_aligned[i] > atr_median
        
        if vol_filter:
            # Long: price breaks above Donchian high AND above EMA50 (uptrend)
            if close[i] > donchian_high[i] and close[i] > ema50[i]:
                signals[i] = 0.25
            # Short: price breaks below Donchian low AND below EMA50 (downtrend)
            elif close[i] < donchian_low[i] and close[i] < ema50[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "4h_Donchian_EMA50_VolatilityFilter"
timeframe = "4h"
leverage = 1.0