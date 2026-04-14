#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and 1d ATR volatility filter
# Donchian(20) captures breakouts at 20-period highs/lows, working in both bull and bear markets
# Volume > 1.3x average confirms breakout strength and reduces false signals
# 1d ATR(14) < 0.03 * price ensures we only trade in low volatility regimes (avoid whipsaws)
# Exit when price crosses the opposite Donchian band (mean reversion within structure)
# Target: 20-40 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ATR(14)
    atr_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR calculation
    atr = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Align ATR to 4h timeframe
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    
    # Donchian Channels (20 periods)
    dc_len = 20
    upper_channel = pd.Series(high).rolling(window=dc_len, min_periods=dc_len).max().values
    lower_channel = pd.Series(low).rolling(window=dc_len, min_periods=dc_len).min().values
    
    # Volume average (20 periods)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, dc_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_aligned[i]) or 
            np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR < 3% of price (low volatility regime)
        low_volatility = atr_aligned[i] < 0.03 * close[i]
        
        # Volume confirmation: current volume > 1.3x average
        volume_confirmed = volume[i] > 1.3 * vol_ma[i]
        
        if position == 0:
            # Enter long: price breaks above upper Donchian + volume + low vol
            if (close[i] > upper_channel[i-1] and 
                volume_confirmed and 
                low_volatility):
                position = 1
                signals[i] = position_size
            # Enter short: price breaks below lower Donchian + volume + low vol
            elif (close[i] < lower_channel[i-1] and 
                  volume_confirmed and 
                  low_volatility):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below lower Donchian (mean reversion)
            if close[i] < lower_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above upper Donchian (mean reversion)
            if close[i] > upper_channel[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Breakout_Volume_ATR_v1"
timeframe = "4h"
leverage = 1.0