#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for HTF analysis (trend, volatility, volume)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d ATR(14) for volatility filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # 1d Volume ratio (current volume / 20-period average)
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume_1d / np.where(vol_ma_20 == 0, 1, vol_ma_20)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_aligned[i]) or 
            np.isnan(vol_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_1d[i]  # Using 1d close for signal generation (but aligned to 1h)
        vol = volume_1d[i] if i < len(volume_1d) else 0
        
        # Get current values
        ema_50 = ema_50_1d_aligned[i]
        atr = atr_14_aligned[i]
        vol_ratio = vol_ratio_aligned[i]
        
        # Volatility filter: only trade when volatility is moderate
        atr_ma_20 = np.nan
        if i >= 20:
            atr_ma_20 = np.mean(atr_14_aligned[i-20:i])
        vol_filter = (atr > 0.5 * atr_ma_20) and (atr < 2.0 * atr_ma_20) if not np.isnan(atr_ma_20) else False
        
        # Volume filter: require above-average volume for confirmation
        vol_filter = vol_filter and (vol_ratio > 1.2)
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        session_filter = 8 <= hour <= 20
        
        if position == 0:
            # Long when price above EMA50 with volume and volatility confirmation
            if (price > ema_50) and vol_filter and session_filter:
                signals[i] = 0.20
                position = 1
            # Short when price below EMA50 with volume and volatility confirmation
            elif (price < ema_50) and vol_filter and session_filter:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA50 or volatility spikes
            if (price < ema_50) or (atr > 2.5 * atr_ma_20 if not np.isnan(atr_ma_20) else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price crosses above EMA50 or volatility spikes
            if (price > ema_50) or (atr > 2.5 * atr_ma_20 if not np.isnan(atr_ma_20) else False):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_EMA50_Trend_VolumeFilter_Session_v1"
timeframe = "1h"
leverage = 1.0