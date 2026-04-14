#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d regime filter
# Elder Ray = Bull Power (High - EMA13), Bear Power (Low - EMA13)
# Bullish when Bull Power > 0 and rising, Bearish when Bear Power < 0 and falling
# 1d regime: use ADX(1d) > 25 to filter for trending markets only
# Works in both bull/bear because we only trade strong trends (ADX filter)
# Exit when power crosses zero (mean reversion within trend)
# Target: 20-40 trades/year per symbol to avoid fee drag

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for ADX regime filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d ADX (14 periods)
    adx_len = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    tr_sum = pd.Series(tr).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_plus_sum = pd.Series(dm_plus).rolling(window=adx_len, min_periods=adx_len).sum().values
    dm_minus_sum = pd.Series(dm_minus).rolling(window=adx_len, min_periods=adx_len).sum().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus_sum / tr_sum
    minus_di = 100 * dm_minus_sum / tr_sum
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx_1d = pd.Series(dx).rolling(window=adx_len, min_periods=adx_len).mean().values
    
    # Align ADX to 6h timeframe
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Elder Ray on 6h: EMA13 of close
    ema_len = 13
    ema = pd.Series(close).ewm(span=ema_len, adjust=False, min_periods=ema_len).mean().values
    bull_power = high - ema  # High - EMA13
    bear_power = low - ema   # Low - EMA13
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, ema_len, adx_len*2)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(adx_1d_aligned[i]) or 
            np.isnan(ema[i]) or
            np.isnan(bull_power[i]) or
            np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Enter long: Bull Power > 0 and rising (bullish momentum) + trending
            if (bull_power[i] > 0 and 
                bull_power[i] > bull_power[i-1] and 
                trending):
                position = 1
                signals[i] = position_size
            # Enter short: Bear Power < 0 and falling (bearish momentum) + trending
            elif (bear_power[i] < 0 and 
                  bear_power[i] < bear_power[i-1] and 
                  trending):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power crosses below zero (loss of bullish momentum)
            if bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Bear Power crosses above zero (loss of bearish momentum)
            if bear_power[i] >= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_Power_1dADXFilter_v1"
timeframe = "6h"
leverage = 1.0