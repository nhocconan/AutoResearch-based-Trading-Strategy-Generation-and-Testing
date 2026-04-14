#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate daily ATR(14) for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = np.abs(high_1d[1:] - low_1d[1:])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d = np.concatenate([[np.nan], tr_1d])
    atr_1d = np.full(len(tr_1d), np.nan)
    
    for i in range(14, len(tr_1d)):
        atr_1d[i] = np.nanmean(tr_1d[i-13:i+1])
    
    # Calculate daily EMA(200) for long-term trend filter
    close_1d_series = pd.Series(close_1d)
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily 50-period high and low for Donchian channel
    high_50_1d = pd.Series(high_1d).rolling(window=50, min_periods=50).max().values
    low_50_1d = pd.Series(low_1d).rolling(window=50, min_periods=50).min().values
    
    # Create arrays for alignment
    atr_1d_arr = atr_1d
    ema_200_1d_arr = ema_200_1d
    high_50_1d_arr = high_50_1d
    low_50_1d_arr = low_50_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(100, n):
        # Get aligned daily data
        atr_1d_i = align_htf_to_ltf(prices, df_1d, atr_1d_arr)[i]
        ema_200_1d_i = align_htf_to_ltf(prices, df_1d, ema_200_1d_arr)[i]
        high_50_1d_i = align_htf_to_ltf(prices, df_1d, high_50_1d_arr)[i]
        low_50_1d_i = align_htf_to_ltf(prices, df_1d, low_50_1d_arr)[i]
        
        if np.isnan(atr_1d_i) or np.isnan(ema_200_1d_i) or \
           np.isnan(high_50_1d_i) or np.isnan(low_50_1d_i):
            continue
        
        # Volatility regime: only trade when ATR is above median (avoid chop)
        if atr_1d_i < np.nanmedian(atr_1d):
            continue
        
        # Only trade in direction of long-term trend (above/below EMA200)
        if close[i] > ema_200_1d_i:
            # Only allow longs in uptrend
            if position == 0:
                # Long: price breaks above daily Donchian high + volume spike
                if close[i] > high_50_1d_i and volume[i] > 2.0 * np.nanmedian(volume[max(0, i-30):i]):
                    position = 1
                    signals[i] = position_size
            elif position == 1:
                # Exit: price crosses below 25-period low
                if close[i] < low_50_1d_i:
                    position = 0
                    signals[i] = 0.0
        elif close[i] < ema_200_1d_i:
            # Only allow shorts in downtrend
            if position == 0:
                # Short: price breaks below daily Donchian low + volume spike
                if close[i] < low_50_1d_i and volume[i] > 2.0 * np.nanmedian(volume[max(0, i-30):i]):
                    position = -1
                    signals[i] = -position_size
            elif position == -1:
                # Exit: price crosses above 25-period high
                if close[i] > high_50_1d_i:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "4h_DailyDonchianBreakout_EMA200_TrendFilter_v2"
timeframe = "4h"
leverage = 1.0