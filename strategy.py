#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_Momentum_Trend_Filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily 50 EMA for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Momentum: 6-period ROC on 6h
    roc_period = 6
    roc = np.zeros(n)
    roc[:roc_period] = np.nan
    roc[roc_period:] = (close[roc_period:] - close[:-roc_period]) / close[:-roc_period] * 100
    
    # Volume confirmation: current volume > 1.5x 20-period average (6h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(roc[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        ema_trend = ema_50_1d_aligned[i]
        mom = roc[i]
        
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long: positive momentum + above daily EMA50 + volume
            if mom > 0 and price > ema_trend and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: negative momentum + below daily EMA50 + volume
            elif mom < 0 and price < ema_trend and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: momentum turns negative or price below daily EMA
            if mom < 0 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: momentum turns positive or price above daily EMA
            if mom > 0 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals