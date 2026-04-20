#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for trend and volatility (1d timeframe)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily ATR for volatility filter (14-period)
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    high_low[0] = high_1d[0] - low_1d[0]
    high_close[0] = np.abs(high_1d[0] - close_1d[0])
    low_close[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate daily volume average (10-period)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=10, min_periods=10).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Load weekly data for longer-term trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly EMA (21-period)
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Calculate weekly high-low range for volatility regime
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    weekly_range = high_1w - low_1w
    # Weekly ATR equivalent (14-period)
    tr_w = np.maximum(high_1w - low_1w, 
                      np.maximum(np.abs(high_1w - np.roll(close_1w, 1)),
                                 np.abs(low_1w - np.roll(close_1w, 1))))
    tr_w[0] = high_1w[0] - low_1w[0]
    atr_w = pd.Series(tr_w).rolling(window=14, min_periods=14).mean().values
    atr_w_aligned = align_htf_to_ltf(prices, df_1w, atr_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_21_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i]) or np.isnan(atr_w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get current 1h bar data
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        
        # Time filter: 08-20 UTC (already converted to datetime64 in index)
        hour = prices.index[i].hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: price above weekly EMA with volume and volatility confirmation
            if (price > ema_21_1w_aligned[i] and 
                vol > 1.5 * vol_ma_1d_aligned[i] and 
                atr_1d_aligned[i] > 0.5 * atr_w_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below weekly EMA with volume and volatility confirmation
            elif (price < ema_21_1w_aligned[i] and 
                  vol > 1.5 * vol_ma_1d_aligned[i] and 
                  atr_1d_aligned[i] > 0.5 * atr_w_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        # Exit conditions
        elif position == 1:
            # Exit long: price crosses below weekly EMA or volatility drops
            if price < ema_21_1w_aligned[i] or vol < 0.6 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price crosses above weekly EMA or volatility drops
            if price > ema_21_1w_aligned[i] or vol < 0.6 * vol_ma_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_WeeklyEMA20_VolumeVolatilityFilter"
timeframe = "1h"
leverage = 1.0