#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    
    # Daily ATR (14-period) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.roll(close_1d, 1))
    low_close = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(high_low, np.maximum(high_close, low_close))
    tr_1d[0] = high_low[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().shift(1).values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Daily EMA200 for trend filter
    ema_200_1d = pd.Series(close_1d).ewm(span=200, min_periods=200, adjust=False).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour Donchian channels (20-period) - use previous bar's high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_4h = high_series.rolling(window=20, min_periods=20).max().shift(1).values
    lower_4h = low_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # 4-hour average volume (20-period) - previous bar
    vol_series = pd.Series(volume)
    avg_vol_4h = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25
    
    start = max(20, 200, 14)
    for i in range(start, n):
        if (np.isnan(upper_4h[i]) or np.isnan(lower_4h[i]) or 
            np.isnan(avg_vol_4h[i]) or np.isnan(ema_200_1d_aligned[i]) or 
            np.isnan(atr_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Additional volatility filter: require current volatility to be above average
        vol_filter = vol > 0.5 * avg_vol_4h[i]  # Reduced from 2.0 to allow more trades
        
        if position == 0:
            # Long: breakout above upper band + volume filter + price above daily EMA200
            if (price > upper_4h[i] and vol_filter and price > ema_200_1d_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: breakout below lower band + volume filter + price below daily EMA200
            elif (price < lower_4h[i] and vol_filter and price < ema_200_1d_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price closes below lower band OR below daily EMA200 OR volatility drops
            if (price < lower_4h[i] or price < ema_200_1d_aligned[i] or 
                vol < 0.3 * avg_vol_4h[i]):  # Exit on low volatility
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price closes above upper band OR above daily EMA200 OR volatility drops
            if (price > upper_4h[i] or price > ema_200_1d_aligned[i] or 
                vol < 0.3 * avg_vol_4h[i]):  # Exit on low volatility
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Donchian_VolumeFilter_EMA200Trend"
timeframe = "4h"
leverage = 1.0