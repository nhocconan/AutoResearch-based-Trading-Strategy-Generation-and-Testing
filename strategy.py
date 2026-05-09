#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h timeframe with 1d HTF. Uses 1d EMA34 for trend, 1d ATR for volatility filter, and 12h price action for entry.
# Combines trend filter with volatility-adjusted breakout to work in both bull and bear markets.
# Target: 50-150 total trades over 4 years with low frequency to minimize fee drag.
name = "12h_EMA34_ATR_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate ATR(14) on 1d for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(np.abs(low_1d[1:] - close_1d[:-1]), tr1)
    tr = np.concatenate([[np.nan], tr2])  # First TR is NaN
    atr14_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate 12h Donchian channels (20-period) for breakout signals
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Need EMA34 and Donchian channels
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34 = ema34_1d_aligned[i]
        atr = atr14_1d_aligned[i]
        upper_break = donchian_high[i]
        lower_break = donchian_low[i]
        
        if position == 0:
            # Enter long: Price breaks above Donchian high + 1d uptrend + volatility filter
            if close[i] > upper_break and close[i] > ema34 and atr > 0:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below Donchian low + 1d downtrend + volatility filter
            elif close[i] < lower_break and close[i] < ema34 and atr > 0:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price breaks below Donchian low or 1d trend turns down
            if close[i] < lower_break or close[i] < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price breaks above Donchian high or 1d trend turns up
            if close[i] > upper_break or close[i] > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals