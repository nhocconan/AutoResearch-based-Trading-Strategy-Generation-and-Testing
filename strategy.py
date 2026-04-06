#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(55) breakout with 1-week EMA(20) trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel and weekly EMA is rising.
# Short when price breaks below lower Donchian channel and weekly EMA is falling.
# Volume must be above 1.5x weekly average for confirmation.
# Designed for 1d timeframe to target 30-100 trades over 4 years with low frequency.

name = "1d_donchian55_1w_ema20_vol_v2"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day Donchian channels (55-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(54, n):
        donchian_high[i] = np.max(high[i-54:i+1])
        donchian_low[i] = np.min(low[i-54:i+1])
    
    # 1-week EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        ema_1w[19] = np.mean(close_1w[:20])
        multiplier = 2 / (20 + 1)
        for i in range(20, len(close_1w)):
            ema_1w[i] = (close_1w[i] - ema_1w[i-1]) * multiplier + ema_1w[i-1]
    
    ema_1w_rising = np.full(len(close_1w), False)
    ema_1w_falling = np.full(len(close_1w), False)
    for i in range(20, len(close_1w)):
        ema_1w_rising[i] = ema_1w[i] > ema_1w[i-1]
        ema_1w_falling[i] = ema_1w[i] < ema_1w[i-1]
    
    ema_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_rising)
    ema_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_falling)
    
    # 1-week volume average for confirmation
    vol_1w = df_1w['volume'].values
    vol_ma_1w = np.full(len(vol_1w), np.nan)
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_ma_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_ma_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(55, 20, 5)  # Donchian needs 55, EMA needs 20, volume needs 5
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_1w_rising_aligned[i]) or np.isnan(ema_1w_falling_aligned[i]) or
            np.isnan(vol_ma_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below lower Donchian or stoploss
            if (close[i] < donchian_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above upper Donchian or stoploss
            if (close[i] > donchian_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            if volume_filter:
                # Long: break above upper Donchian with rising weekly EMA
                if (close[i] > donchian_high[i] and ema_1w_rising_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below lower Donchian with falling weekly EMA
                elif (close[i] < donchian_low[i] and ema_1w_falling_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals