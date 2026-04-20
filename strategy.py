#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data ONCE for weekly indicators
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly 10-period EMA for trend filter
    ema_10w = pd.Series(close_1w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate weekly ATR for volatility filter
    high_w = high_1w
    low_w = low_1w
    close_w = close_1w
    tr1_w = high_w[1:] - low_w[1:]
    tr2_w = np.abs(high_w[1:] - close_w[:-1])
    tr3_w = np.abs(low_w[1:] - close_w[:-1])
    tr_w = np.concatenate([[np.nan], np.maximum(tr1_w, np.maximum(tr2_w, tr3_w))])
    atr_10w = pd.Series(tr_w).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Align weekly indicators to daily timeframe
    ema_10w_aligned = align_htf_to_ltf(prices, df_1w, ema_10w)
    atr_10w_aligned = align_htf_to_ltf(prices, df_1w, atr_10w)
    
    # Calculate daily 20-period Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 1d timeframe (no additional alignment needed as we're already on 1d)
    # But we still use the helper for consistency
    highest_20d_aligned = align_htf_to_ltf(prices, df_1d, highest_20d)
    lowest_20d_aligned = align_htf_to_ltf(prices, df_1d, lowest_20d)
    
    # Calculate daily ATR for stop sizing
    high_d = high_1d
    low_d = low_1d
    close_d = close_1d
    tr1_d = high_d[1:] - low_d[1:]
    tr2_d = np.abs(high_d[1:] - close_d[:-1])
    tr3_d = np.abs(low_d[1:] - close_d[:-1])
    tr_d = np.concatenate([[np.nan], np.maximum(tr1_d, np.maximum(tr2_d, tr3_d))])
    atr_14d = pd.Series(tr_d).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: daily volume > 20-period average
    volume_d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if NaN in indicators
        if (np.isnan(ema_10w_aligned[i]) or np.isnan(atr_10w_aligned[i]) or 
            np.isnan(highest_20d_aligned[i]) or np.isnan(lowest_20d_aligned[i]) or 
            np.isnan(atr_14d[i]) or np.isnan(volume_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Price levels
        price = close_1d[i]
        weekly_trend = ema_10w_aligned[i]
        weekly_vol = atr_10w_aligned[i]
        resistance = highest_20d_aligned[i]
        support = lowest_20d_aligned[i]
        daily_vol = atr_14d[i]
        vol_filter = volume_d[i] > volume_ma_20[i]
        
        if position == 0:
            # Long: price breaks above weekly EMA (trend filter) AND above 20-day resistance, with volume
            if price > weekly_trend and price > resistance and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below weekly EMA (trend filter) AND below 20-day support, with volume
            elif price < weekly_trend and price < support and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: stop loss (2x ATR below entry) or price breaks below 20-day support
            if price <= entry_price - 2.0 * daily_vol or price < support:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: stop loss (2x ATR above entry) or price breaks above 20-day resistance
            if price >= entry_price + 2.0 * daily_vol or price > resistance:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyEMA10_Donchian20_VolumeFilter"
timeframe = "1d"
leverage = 1.0