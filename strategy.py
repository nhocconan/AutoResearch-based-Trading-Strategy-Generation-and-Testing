#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly and daily data for multi-timeframe analysis
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # Weekly close for long-term trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily ATR for volatility regime
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
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
    
    # 12h volume filter (current / 30-period average)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_ratio = volume / np.where(vol_ma_30 == 0, 1, vol_ma_30)
    
    # 12h Donchian(20) channels
    high_12h = pd.Series(close).rolling(window=20, min_periods=20).max().values
    low_12h = pd.Series(close).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if NaN in critical values
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(high_12h[i]) or np.isnan(low_12h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        ema_trend = ema_50_1w_aligned[i]
        atr = atr_14_1d_aligned[i]
        vol_ratio_12h = vol_ratio[i]
        donch_high = high_12h[i]
        donch_low = low_12h[i]
        
        # Long-term trend filter: price above/below weekly EMA50
        trend_up = price > ema_trend
        trend_down = price < ema_trend
        
        # Volatility filter: avoid low volatility (chop)
        atr_ma_30 = pd.Series(atr_14_1d_aligned).rolling(window=30, min_periods=30).mean().values[i]
        vol_filter = atr > 0.5 * atr_ma_30
        
        # Volume filter: require above-average volume
        vol_filter = vol_filter and (vol_ratio_12h > 1.5)
        
        if position == 0:
            # Enter long on Donchian breakout with volume and trend
            if price > donch_high and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short on Donchian breakdown with volume and trend
            elif price < donch_low and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retracement to midpoint or volatility spike
            midpoint = (donch_high + donch_low) / 2
            if price < midpoint or atr > 4.0 * atr_ma_30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retracement to midpoint or volatility spike
            midpoint = (donch_high + donch_low) / 2
            if price > midpoint or atr > 4.0 * atr_ma_30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_WeeklyTrend_Volume_Filter_v1"
timeframe = "12h"
leverage = 1.0