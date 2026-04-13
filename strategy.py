#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian(20) breakout + volume confirmation + ATR filter
    # Long when: price breaks above 20-bar high AND volume > 1.5x 20-bar avg volume AND ATR(14) > 0.5 * ATR(50)
    # Short when: price breaks below 20-bar low AND volume > 1.5x 20-bar avg volume AND ATR(14) > 0.5 * ATR(50)
    # Exit when: price crosses 10-bar EMA (fast exit) OR ATR(14) < 0.3 * ATR(50) (low volatility regime)
    # Uses discrete sizing (0.25) targeting 75-150 total trades over 4 years.
    # Donchian provides clear breakout levels; volume confirms institutional participation;
    # ATR ratio filter ensures trading only in sufficient volatility environments.
    # Works in bull (breakouts continue) and bear (strong breakdowns only).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    
    # Calculate 10-bar EMA for exit
    close_series = pd.Series(close)
    ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate ATR(14) and ATR(50) for volatility filter
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.concatenate([[close[0]], close[:-1]]))
        tr3 = np.abs(low - np.concatenate([[close[0]], close[:-1]]))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        
        # Wilder's ATR
        atr = np.full_like(tr, np.nan, dtype=float)
        if len(tr) >= period:
            atr[period-1] = np.mean(tr[:period])
            for i in range(period, len(tr)):
                atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
        return atr
    
    atr_14 = calculate_atr(high, low, close, 14)
    atr_50 = calculate_atr(high, low, close, 50)
    
    # Volume confirmation: volume > 1.5x 20-bar average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * avg_volume)
    
    # ATR filter: trade only when short-term ATR is sufficiently above long-term ATR
    atr_ratio = atr_14 / atr_50
    volatility_filter = atr_ratio > 0.5  # sufficient volatility
    low_volatility_exit = atr_ratio < 0.3  # exit when volatility drops
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):  # start after ATR(50) warmup
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_10[i]) or np.isnan(atr_ratio[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions (using previous bar's levels to avoid look-ahead)
        breakout_up = close[i] > donchian_high[i-1]  # break above previous 20-bar high
        breakout_down = close[i] < donchian_low[i-1]  # break below previous 20-bar low
        
        # Entry conditions with volume and volatility filters
        long_entry = breakout_up and volume_confirmed[i] and volatility_filter[i] and position != 1
        short_entry = breakout_down and volume_confirmed[i] and volatility_filter[i] and position != -1
        
        # Exit conditions
        exit_long = (position == 1 and (close[i] < ema_10[i] or low_volatility_exit[i]))
        exit_short = (position == -1 and (close[i] > ema_10[i] or low_volatility_exit[i]))
        
        # Execute signals
        if long_entry:
            position = 1
            signals[i] = position_size
        elif short_entry:
            position = -1
            signals[i] = -position_size
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_donchian_breakout_volume_atr_filter_v1"
timeframe = "4h"
leverage = 1.0