#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Time filter: 08-20 UTC
    hour = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hour >= 8) & (hour <= 20)
    
    # Get 4h data for trend
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA(34) for trend
    close_4h_series = pd.Series(close_4h)
    ema_4h = close_4h_series.ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Get 1d data for Donchian channel
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d Donchian(20) breakout levels
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    donch_high = high_1d_series.rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = low_1d_series.rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align Donchian levels to 1h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 40  # for Donchian and volume calculations
    
    for i in range(start, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or
            np.isnan(ema_4h_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        trend = ema_4h_aligned[i]
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume confirmation AND above 4h EMA
            if price > donch_high_aligned[i] and vol > 1.5 * avg_vol[i] and price > trend:
                position = 1
                signals[i] = position_size
            # Short: price breaks below 1d Donchian low with volume confirmation AND below 4h EMA
            elif price < donch_low_aligned[i] and vol > 1.5 * avg_vol[i] and price < trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below 1d Donchian low OR below 4h EMA
            if price < donch_low_aligned[i] or price < trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above 1d Donchian high OR above 4h EMA
            if price > donch_high_aligned[i] or price > trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_Donchian_EMA_Volume_Filter"
timeframe = "1h"
leverage = 1.0