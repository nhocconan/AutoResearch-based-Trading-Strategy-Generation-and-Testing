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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14)
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
    
    # Calculate daily EMA(50) and EMA(200)
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_1d = close_1d_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate daily ADX(14) for trend strength
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    tr_14 = np.zeros_like(tr_1d)
    tr_14[14:] = [np.nanmean(tr_1d[i-13:i+1]) for i in range(14, len(tr_1d))]
    
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / tr_14
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / tr_14
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    adx = np.concatenate([np.full(27, np.nan), adx[27:]])  # Align length
    
    # Calculate daily median volume for volume filter
    vol_1d = df_1d['volume'].values
    median_vol_1d = np.nanmedian(vol_1d)
    
    # Create arrays for alignment
    atr_1d_arr = atr_1d
    ema_50_1d_arr = ema_50_1d
    ema_200_1d_arr = ema_200_1d
    adx_1d_arr = adx
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25
    
    for i in range(50, n):
        # Get aligned daily data
        atr_1d_i = align_htf_to_ltf(prices, df_1d, atr_1d_arr)[i]
        ema_50_1d_i = align_htf_to_ltf(prices, df_1d, ema_50_1d_arr)[i]
        ema_200_1d_i = align_htf_to_ltf(prices, df_1d, ema_200_1d_arr)[i]
        adx_1d_i = align_htf_to_ltf(prices, df_1d, adx_1d_arr)[i]
        
        if np.isnan(atr_1d_i) or np.isnan(ema_50_1d_i) or np.isnan(ema_200_1d_i) or np.isnan(adx_1d_i):
            continue
        
        # Only trade when trend is strong (ADX > 25) and volatility is elevated
        if adx_1d_i < 25 or atr_1d_i < np.nanmedian(atr_1d):
            continue
        
        # Only trade in direction of long-term trend (above/below EMA200)
        if close[i] > ema_200_1d_i:
            # Only allow longs in uptrend
            if position == 0:
                # Long: price above daily EMA50 + volume spike
                if close[i] > ema_50_1d_i and volume[i] > 2.0 * median_vol_1d:
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
                if close[i] < ema_50_1d_i and volume[i] > 2.0 * median_vol_1d:
                    position = -1
                    signals[i] = -position_size
            elif position == -1:
                # Exit: price crosses above EMA50
                if close[i] > ema_50_1d_i:
                    position = 0
                    signals[i] = 0.0
    
    return signals

name = "4h_DailyEMA50_ADX25_Volume2x_TrendFilter"
timeframe = "4h"
leverage = 1.0