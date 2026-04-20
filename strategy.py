#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load 4h data for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA200 for long-term trend
    ema_200_4h = pd.Series(close_4h).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 4h ADX for trend strength
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_arr = df_4h['close'].values
    
    # Calculate TR
    high_low = high_4h[1:] - high_4h[:-1]
    high_close = np.abs(high_4h[1:] - np.roll(close_4h_arr, 1)[1:])
    low_close = np.abs(low_4h[1:] - np.roll(close_4h_arr, 1)[1:])
    high_low = np.concatenate([[high_4h[0] - low_4h[0]], high_low])
    high_close = np.concatenate([[np.abs(high_4h[0] - close_4h_arr[0])], high_close])
    low_close = np.concatenate([[np.abs(low_4h[0] - close_4h_arr[0])], low_close])
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Calculate DM
    up_move = high_4h[1:] - high_4h[:-1]
    down_move = low_4h[:-1] - low_4h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Calculate DI and ADX
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    plus_di_14 = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / (tr_14 + 1e-10)
    minus_di_14 = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / (tr_14 + 1e-10)
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_4h = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_4h_aligned = align_htf_to_ltf(prices, df_4h, adx_4h)
    
    # Load daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Skip if NaN in critical values
        if (np.isnan(ema_200_4h_aligned[i]) or np.isnan(adx_4h_aligned[i]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        vol = prices['volume'].iloc[i]
        
        if position == 0:
            # Long: price above 4h EMA200, strong trend (ADX > 25), volume confirmation
            if (price > ema_200_4h_aligned[i] and 
                adx_4h_aligned[i] > 25 and 
                vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price below 4h EMA200, strong trend (ADX > 25), volume confirmation
            elif (price < ema_200_4h_aligned[i] and 
                  adx_4h_aligned[i] > 25 and 
                  vol > 1.5 * vol_ma_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 4h EMA200 or trend weakens (ADX < 20)
            if price < ema_200_4h_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above 4h EMA200 or trend weakens (ADX < 20)
            if price > ema_200_4h_aligned[i] or adx_4h_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_EMA200_ADX25_VolumeFilter_Session"
timeframe = "1h"
leverage = 1.0