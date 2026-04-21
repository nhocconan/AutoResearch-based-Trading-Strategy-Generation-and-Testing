#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data ONCE before loop for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # 4h EMA21 for trend filter
    close_4h = df_4h['close'].values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # 1h volume confirmation: volume / 20-period average volume
    vol_ma_20 = pd.Series(prices['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1h = prices['volume'].values / vol_ma_20
    
    # Pre-compute session filter (8-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_mask = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready or outside session
        if np.isnan(ema_21_4h_aligned[i]) or np.isnan(vol_ratio_1h[i]) or not session_mask[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        ema_trend = ema_21_4h_aligned[i]
        vol_ratio = vol_ratio_1h[i]
        
        if position == 0:
            # Enter long: price above EMA21 (uptrend) with volume spike
            if (price_close > ema_trend and vol_ratio > 1.5):
                signals[i] = 0.20
                position = 1
            # Enter short: price below EMA21 (downtrend) with volume spike
            elif (price_close < ema_trend and vol_ratio > 1.5):
                signals[i] = -0.20
                position = -1
        
        elif position != 0:
            # Exit: reverse trend or low volume
            if position == 1 and (price_close < ema_trend or vol_ratio < 0.8):
                signals[i] = 0.0
                position = 0
            elif position == -1 and (price_close > ema_trend or vol_ratio < 0.8):
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.20 if position == 1 else -0.20
    
    return signals

name = "1h_EMA21_Volume_Spike_Session"
timeframe = "1h"
leverage = 1.0