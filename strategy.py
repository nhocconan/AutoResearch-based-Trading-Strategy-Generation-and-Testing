#!/usr/bin/env python3
"""
4h Donchian Breakout + 1d Trend + Volume Confirmation + ATR Stop
Long: Close > Donchian High(20) + 1d EMA(50) rising + Volume > 1.5x 4h Volume SMA(20)
Short: Close < Donchian Low(20) + 1d EMA(50) falling + Volume > 1.5x 4h Volume SMA(20)
Exit: Opposite Donchian break or close below/above ATR trailing stop
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(50) for trend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h Donchian Channel (20)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h Volume SMA(20)
    volume_series = pd.Series(volume)
    volume_sma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Calculate ATR(14) for stop loss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = max(20, 50)  # need Donchian and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_sma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = volume_sma_20[i]
        ema_trend = ema_50_1d_aligned[i]
        ema_prev = ema_50_1d_aligned[i-1]
        d_high = donchian_high[i]
        d_low = donchian_low[i]
        atr_val = atr[i]
        
        if position == 0:
            # Long: Break above Donchian high + 1d EMA rising + volume spike
            if price > d_high and ema_trend > ema_prev and vol > 1.5 * vol_sma_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = price - 2.0 * atr_val
            # Short: Break below Donchian low + 1d EMA falling + volume spike
            elif price < d_low and ema_trend < ema_prev and vol > 1.5 * vol_sma_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = price + 2.0 * atr_val
        
        elif position == 1:
            # Long exit: Break below Donchian low OR price hits ATR stop
            if price < d_low or price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop up
                atr_stop = max(atr_stop, price - 2.0 * atr_val)
        
        elif position == -1:
            # Short exit: Break above Donchian high OR price hits ATR stop
            if price > d_high or price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop down
                atr_stop = min(atr_stop, price + 2.0 * atr_val)
    
    return signals

name = "4h_Donchian_Breakout_1dEMA50_VolumeSpike_ATRStop"
timeframe = "4h"
leverage = 1.0