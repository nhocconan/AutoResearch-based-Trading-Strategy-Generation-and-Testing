#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 12h data for trend and volatility
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(34) for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h ATR(14) for volatility and stop
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr2[0] = tr1[0]
    tr3[0] = tr1[0]
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_14_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_14_12h)
    
    # 12h Volume ratio (current / 20-period average)
    vol_ma_20_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    vol_ratio_12h = volume_12h / np.where(vol_ma_20_12h == 0, 1, vol_ma_20_12h)
    vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    
    # Load 4h price data (primary timeframe)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        atr = atr_14_12h_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        
        # Trend filter: price must be above/below EMA
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volatility filter: moderate volatility only
        atr_ma_20 = pd.Series(atr_14_12h_aligned).rolling(window=20, min_periods=20).mean().values[i]
        vol_filter = (atr > 0.3 * atr_ma_20) and (atr < 3.0 * atr_ma_20)
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio > 1.3)
        
        if position == 0:
            # Long when uptrend + volume confirmation
            if trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short when downtrend + volume confirmation
            elif trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend reversal or volatility spike
            if not trend_up or (atr > 4.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reversal or volatility spike
            if not trend_down or (atr > 4.0 * atr_ma_20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_EMA34_VolumeTrend_Filter_v1"
timeframe = "4h"
leverage = 1.0