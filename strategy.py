#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for multi-timeframe analysis
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # 4h EMA trend filter (21-period)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d ATR for volatility filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first value
    atr_1d = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # 1h Donchian channel breakout (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    # Start after enough data for calculations
    start = 50  # ensures 20-period indicators have values
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]) or
            np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Determine market regime based on 4h EMA slope
        if i >= start + 1:
            ema_prev = ema_4h_aligned[i-1]
            ema_curr = ema_4h_aligned[i]
            ema_slope = ema_curr - ema_prev
            # Strong trend: |slope| > 0.5 * ATR (scaled to 1h)
            strong_trend = abs(ema_slope) > 0.5 * (atr_1d_aligned[i] / 6.0)  # ATR scaled to hourly
        else:
            strong_trend = False
        
        if position == 0:
            # Long: price breaks above Donchian high + volume + strong uptrend
            if (price > high_max[i] and vol > 1.5 * vol_ma[i] and 
                ema_slope > 0 and strong_trend and price > ema_4h_aligned[i]):
                position = 1
                signals[i] = position_size
            # Short: price breaks below Donchian low + volume + strong downtrend
            elif (price < low_min[i] and vol > 1.5 * vol_ma[i] and 
                  ema_slope < 0 and strong_trend and price < ema_4h_aligned[i]):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reversal
            if price < low_min[i] or ema_slope < 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reversal
            if price > high_max[i] or ema_slope > 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4d_Donchian_EMA_Trend_Filter"
timeframe = "1h"
leverage = 1.0