#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d ATR-based volatility regime filter + Donchian(20) breakout with volume confirmation
# Long when price breaks above Donchian(20) high AND ATR(14) > ATR(50) (high volatility regime) AND volume > 1.5 * avg_volume(20)
# Short when price breaks below Donchian(20) low AND ATR(14) > ATR(50) (high volatility regime) AND volume > 1.5 * avg_volume(20)
# Exit when price crosses Donchian(20) midpoint OR ATR(14) < ATR(50) (low volatility regime)
# Uses discrete sizing 0.25 to balance return and risk
# Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe
# ATR regime filter ensures we only trade in high volatility environments where breakouts are more reliable
# Volume confirmation reduces false breakouts
# Works in bull markets (strong breakouts with volume) and bear markets (strong breakdowns with volume)

name = "4h_ATR_Regime_Donchian20_Breakout_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:  # Need enough for ATR(50)
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14) and ATR(50)
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr14_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr50_1d = pd.Series(tr_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    high_vol_regime = atr14_1d_aligned > atr50_1d_aligned  # High volatility when short ATR > long ATR
    
    # Calculate Donchian(20) on 4h data
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    midpoint_20 = (highest_20 + lowest_20) / 2.0
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(midpoint_20[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian(20) high, high volatility regime, volume confirmation, in session
            if close[i] > highest_20[i] and high_vol_regime[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian(20) low, high volatility regime, volume confirmation, in session
            elif close[i] < lowest_20[i] and high_vol_regime[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below Donchian(20) midpoint OR low volatility regime
            if close[i] < midpoint_20[i] or not high_vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above Donchian(20) midpoint OR low volatility regime
            if close[i] > midpoint_20[i] or not high_vol_regime[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals