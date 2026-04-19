#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_1d_PriceActionConfluence_V1"
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
    
    # Get weekly data once before loop
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Get daily data once before loop
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Weekly trend: EMA21 of weekly close
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Daily ATR for volatility filter
    tr1 = np.maximum(high_1d[1:], close_1d[:-1]) - np.minimum(low_1d[1:], close_1d[:-1])
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align weekly EMA and daily ATR/volume to 12h timeframe
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema21_1w_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Conditions
        weekly_uptrend = price > ema21_1w_aligned[i]
        weekly_downtrend = price < ema21_1w_aligned[i]
        low_volatility = atr_1d_aligned[i] < np.nanmedian(atr_1d_aligned[max(0, i-50):i+1])
        high_volume = vol > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long: weekly uptrend + low volatility + high volume
            if weekly_uptrend and low_volatility and high_volume:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend + low volatility + high volume
            elif weekly_downtrend and low_volatility and high_volume:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: weekly trend turns down OR volatility spikes
            if not weekly_uptrend or atr_1d_aligned[i] > 1.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: weekly trend turns up OR volatility spikes
            if not weekly_downtrend or atr_1d_aligned[i] > 1.5 * np.nanmedian(atr_1d_aligned[max(0, i-50):i+1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals