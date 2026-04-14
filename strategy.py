#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate 30-week EMA for trend (weekly)
    ema_30_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 30:
        ema_30_1w[29] = np.mean(close_1w[:30])
        for i in range(30, len(close_1w)):
            ema_30_1w[i] = (close_1w[i] * 2 + ema_30_1w[i-1] * 28) / 30
    
    # Calculate 30-week SMA for volume (weekly)
    sma_vol_30_1w = np.full_like(volume_1w, np.nan)
    if len(volume_1w) >= 30:
        for i in range(29, len(volume_1w)):
            sma_vol_30_1w[i] = np.mean(volume_1w[i-29:i+1])
    
    # Align weekly indicators to daily
    ema_30_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_30_1w)
    sma_vol_30_1w_aligned = align_htf_to_ltf(prices, df_1w, sma_vol_30_1w)
    
    # Calculate daily ATR(14) for volatility filter
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14 = np.full_like(close, np.nan)
    if len(close) >= 14:
        atr_14[13] = np.mean(tr[1:15])
        for i in range(15, len(close)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_30_1w_aligned[i]) or 
            np.isnan(sma_vol_30_1w_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current daily volume vs 30-week average volume
        if sma_vol_30_1w_aligned[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / sma_vol_30_1w_aligned[i]
        
        if position == 0:
            # Long: Price above weekly EMA30 + volume surge
            if (close[i] > ema_30_1w_aligned[i] and
                volume_ratio > 3.0):
                position = 1
                signals[i] = position_size
            # Short: Price below weekly EMA30 + volume surge
            elif (close[i] < ema_30_1w_aligned[i] and
                  volume_ratio > 3.0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price crosses below weekly EMA30 OR volume drops
            if (close[i] < ema_30_1w_aligned[i] or 
                volume_ratio < 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price crosses above weekly EMA30 OR volume drops
            if (close[i] > ema_30_1w_aligned[i] or 
                volume_ratio < 1.5):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_EMA30_VolumeSurge"
timeframe = "1d"
leverage = 1.0