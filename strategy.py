#!/usr/bin/env python3
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
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
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
    
    # Calculate daily EMA(50) for trend
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily EMA(200) for long-term trend filter
    close_1d_series2 = pd.Series(close_1d)
    ema_200_1d = close_1d_series2.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Create arrays for alignment
    atr_1d_arr = atr_1d
    ema_50_1d_arr = ema_50_1d
    ema_200_1d_arr = ema_200_1d
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily data
        atr_1d_i = align_htf_to_ltf(prices, df_1d, atr_1d_arr)[i]
        ema_50_1d_i = align_htf_to_ltf(prices, df_1d, ema_50_1d_arr)[i]
        ema_200_1d_i = align_htf_to_ltf(prices, df_1d, ema_200_1d_arr)[i]
        
        if np.isnan(atr_1d_i) or np.isnan(ema_50_1d_i) or np.isnan(ema_200_1d_i):
            continue
        
        # Volatility regime: only trade when ATR is above median (avoid chop)
        if atr_1d_i < np.nanmedian(atr_1d):
            continue
        
        # Only trade in direction of long-term trend (above/below EMA200)
        if close[i] > ema_200_1d_i:
            # Only allow longs in uptrend
            if position == 0:
                # Long: price above daily EMA50 + volume spike
                if close[i] > ema_50_1d_i and volume[i] > 1.5 * np.nanmedian(volume[max(0, i-20):i]):
                    position = 1
                    signals[i] = position_size
            elif position == 1:
                # Exit: price crosses below EMA50
                if close[i] < ema_50_1d_i:
                    position = 0
                    signals[i] = 0.0
        elif close[i] < ema_200_1d_i:
            # Only allow shorts in downtrend
            if position == 0:
                # Short: price below daily EMA50 + volume spike
                if close[i] < ema_50_1d_i and volume[i] > 1.5 * np.nanmedian(volume[max(0, i-20):i]):
                    position = -1
                    signals[i] = -position_size
            elif position == -1:
                # Exit: price crosses above EMA50
                if close[i] > ema_50_1d_i:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "6h_DailyEMA50_VolumeSpike_TrendFilter_v1"
timeframe = "6h"
leverage = 1.0