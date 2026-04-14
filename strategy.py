#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data for primary timeframe (price)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Get 1d data for HTF context (trend and pivots)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 12h EMA(34) for trend - using close prices
    close_12h_series = pd.Series(close_12h)
    ema_12h = close_12h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate 1d EMA(50) for HTF trend filter
    close_1d_series = pd.Series(close_1d)
    ema_1d = close_1d_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 12h ATR(14) for volatility and stop loss
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_12h_aligned = align_htf_to_ltf(prices, df_12h, atr_12h)
    
    # Calculate 12h Donchian channels (20-period)
    high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    dc_high_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    dc_low_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for EMA and ATR calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(dc_high_aligned[i]) or np.isnan(dc_low_aligned[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(ema_1d_aligned[i]) or
            np.isnan(atr_12h_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        trend_12h = ema_12h_aligned[i]
        trend_1d = ema_1d_aligned[i]
        dc_high = dc_high_aligned[i]
        dc_low = dc_low_aligned[i]
        atr = atr_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume confirmation AND aligned trends
            if price > dc_high and vol > 1.5 * avg_vol[i] and trend_12h > trend_1d:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume confirmation AND aligned trends
            elif price < dc_low and vol > 1.5 * avg_vol[i] and trend_12h < trend_1d:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low OR trend reversal
            if price < dc_low or trend_12h < trend_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high OR trend reversal
            if price > dc_high or trend_12h > trend_1d:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_Donchian_Trend_Volume"
timeframe = "12h"
leverage = 1.0