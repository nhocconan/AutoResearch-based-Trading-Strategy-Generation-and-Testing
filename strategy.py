#!/usr/bin/env python3
"""
Hypothesis: 1d price closes above/below 1-week EMA(20) with 1d volume > 1.5x 20-day average and 
1-week ATR ratio < 0.8 (low volatility regime). Uses weekly trend filter to avoid counter-trend 
trades, volume to confirm conviction, and low volatility to avoid false breakouts. 
Targets 30-80 total trades over 4 years (7-20/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for price and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d EMA(20) for short-term trend
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1d volume spike (volume > 1.5x 20-day average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_1d > (vol_ma_20 * 1.5)
    
    # Get 1w data for trend filter and volatility regime
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate 1-week EMA(20) for trend filter
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 1-week ATR for volatility regime
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                           np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_1w = pd.Series(tr_1w).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1-week ATR ratio (current ATR / 50-period average ATR) for volatility regime
    atr_ma_50 = pd.Series(atr_1w).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_1w / atr_ma_50
    
    # Volatility regime: ATR ratio < 0.8 = low volatility (good for trend following)
    low_volatility = atr_ratio < 0.8
    
    # Align all 1d and 1w data to 1d timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike.astype(float))
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    low_volatility_aligned = align_htf_to_ltf(prices, df_1w, low_volatility.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% of capital
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(vol_spike_aligned[i]) or 
            np.isnan(ema_20_1w_aligned[i]) or 
            np.isnan(low_volatility_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: price vs 1d EMA(20) + volume spike + low volatility + weekly trend filter
        price_above_ema = close[i] > ema_20_1d_aligned[i]
        price_below_ema = close[i] < ema_20_1d_aligned[i]
        vol_confirm = vol_spike_aligned[i] > 0.5  # True if volume spike
        vol_regime = low_volatility_aligned[i] > 0.5  # True if low volatility
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]  # Price above weekly EMA
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]  # Price below weekly EMA
        
        long_entry = price_above_ema and vol_confirm and vol_regime and weekly_uptrend
        short_entry = price_below_ema and vol_confirm and vol_regime and weekly_downtrend
        
        # Exit when price crosses back below/above 1d EMA(20)
        exit_long = position == 1 and close[i] < ema_20_1d_aligned[i]
        exit_short = position == -1 and close[i] > ema_20_1d_aligned[i]
        
        # Execute signals
        if long_entry and position != 1:
            position = 1
            signals[i] = position_size
        elif short_entry and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long or exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_1w_ema_volume_volatility"
timeframe = "1d"
leverage = 1.0