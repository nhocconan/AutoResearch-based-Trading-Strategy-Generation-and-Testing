#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy with 4h Donchian channel trend filter and 1d volume confirmation.
# Uses 4h Donchian(20) breakout for trend direction (long when price > upper band, short when < lower band).
# Entry timing on 1h: only enter when price pulls back to the 4h 20-period EMA in the direction of trend.
# 1d volume filter: current volume > 1.5x 20-period average to avoid low-volume false breakouts.
# Designed for 1h timeframe: trend from 4h, entry timing on 1h, volume filter from 1d.
# Target: 60-150 total trades over 4 years (15-38/year).

name = "1h_donchian20_4h_ema_1d_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period high/low)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 4h EMA(20) for pullback entry
    close_4h = df_4h['close'].values
    close_4h_series = pd.Series(close_4h)
    ema_20_4h = close_4h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d volume filter: current volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    volume_1d_series = pd.Series(volume_1d)
    vol_ma_1d = volume_1d_series.rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if 4h or 1d data not available
        if np.isnan(donchian_high_aligned[i]) or np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        if position == 1:  # long position
            # Exit: price breaks below 4h Donchian low or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < donchian_low_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price breaks above 4h Donchian high or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > donchian_high_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries with volume filter and session
            if volume_filter:
                # Long: price above 4h Donchian high and pulling back to 4h EMA20
                if (close[i] > donchian_high_aligned[i] and 
                    close[i] <= ema_20_4h_aligned[i] * 1.005):  # within 0.5% of EMA
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: price below 4h Donchian low and pulling back to 4h EMA20
                elif (close[i] < donchian_low_aligned[i] and 
                      close[i] >= ema_20_4h_aligned[i] * 0.995):  # within 0.5% of EMA
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
    
    return signals