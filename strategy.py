#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load daily data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    
    # Daily EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily ATR(14) for volatility and stop
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # 12h price and volume data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h volume ratio (current / 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    
    # 12h price channel (Donchian 10)
    high_10 = pd.Series(close).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(close).rolling(window=10, min_periods=10).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):
        # Skip if NaN in critical values
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1d_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_12h = vol_ratio[i]
        
        # Trend filter: price above/below EMA50
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volatility filter: avoid extreme volatility
        atr_ma_20 = pd.Series(atr_14_1d_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_20) and (atr < 2.5 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_12h > 1.4)
        
        if position == 0:
            # Enter long on breakout above Donchian high with trend and volume
            if price > high_10[i] and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on breakdown below Donchian low with trend and volume
            elif price < low_10[i] and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: breakdown below Donchian low or volatility spike
            if price < low_10[i] or atr > 3.0 * atr_ma_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: breakout above Donchian high or volatility spike
            if price > high_10[i] or atr > 3.0 * atr_ma_20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian10_EMA50_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0