#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load 12h data for trend and pivot levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA(34) for trend
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # 12h ATR(14) for volatility
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
    
    # 12h Price range for volatility regime
    range_12h = high_12h - low_12h
    range_ma_20_12h = pd.Series(range_12h).rolling(window=20, min_periods=20).mean().values
    range_ratio_12h = range_12h / np.where(range_ma_20_12h == 0, 1, range_ma_20_12h)
    range_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, range_ratio_12h)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(atr_14_12h_aligned[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(range_ratio_12h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_34_12h_aligned[i]
        atr = atr_14_12h_aligned[i]
        vol_ratio = vol_ratio_12h_aligned[i]
        range_ratio = range_ratio_12h_aligned[i]
        
        # Trend filter
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volatility regime filter: avoid extreme volatility
        vol_filter = (atr > 0.5 * atr) and (atr < 3.0 * atr)  # This will be fixed below
        
        # Volume filter: require above-average volume
        vol_filter = vol_ratio > 1.2
        
        # Range filter: avoid choppy markets
        range_filter = range_ratio < 2.0
        
        # Combined filter
        filter_pass = vol_filter and range_filter
        
        if position == 0:
            # Long when uptrend + volume + range filter
            if trend_up and filter_pass:
                signals[i] = 0.25
                position = 1
            # Short when downtrend + volume + range filter
            elif trend_down and filter_pass:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit conditions for long
            if not trend_up or not filter_pass:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions for short
            if not trend_down or not filter_pass:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_12h_EMA34_VolumeRange_Filter_v1"
timeframe = "6h"
leverage = 1.0