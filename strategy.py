#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian Breakout + Volume Spike + 12h EMA Trend Filter
# Uses Donchian channel breakouts with volume confirmation and 12h EMA trend filter
# 12h EMA provides smoother trend direction than ADX, reducing whipsaws in choppy markets
# Works in bull/bear by capturing breakouts in the direction of the trend
# Target: 75-200 total trades over 4 years (19-50/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channel (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume confirmation: volume > 2x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    # Load 12h EMA once before loop (trend filter)
    df_12h = get_htf_data(prices, '12h')
    ema_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 40  # for Donchian and volume calculations
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(avg_vol[i]) or np.isnan(ema_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: only trade when price is above/below 12h EMA
        if ema_12h_aligned[i] == 0:  # Avoid division by zero
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long: price breaks above Donchian high with volume filter and above 12h EMA
            if price > donchian_high[i] and vol > 2.0 * avg_vol[i] and price > ema_12h_aligned[i]:
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low with volume filter and below 12h EMA
            elif price < donchian_low[i] and vol > 2.0 * avg_vol[i] and price < ema_12h_aligned[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if price < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if price > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_Donchian_Volume_12hEMA_Filter"
timeframe = "4h"
leverage = 1.0