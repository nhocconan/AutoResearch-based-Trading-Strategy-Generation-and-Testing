#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above upper Donchian channel and 1w EMA50 is rising with volume > 1.5x average.
Short when price breaks below lower Donchian channel and 1w EMA50 is falling with volume > 1.5x average.
Exit on opposite Donchian break or EMA50 flattening.
Donchian channels provide structure based on 20-day high/low.
1w EMA50 > rising/falling filters for strong weekly trend to avoid false breakouts in chop.
Volume confirmation ensures breakout legitimacy.
Designed for 1d timeframe targeting 30-100 total trades over 4 years with low frequency to minimize fee drag.
Works in both bull and bear markets by only taking breakouts in direction of strong weekly trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 1d timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian channels from prior 20 days (using 1d data)
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(lookback, n):
        upper_channel[i] = np.max(high[i-lookback:i])
        lower_channel[i] = np.min(low[i-lookback:i])
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_aligned[i]) or np.isnan(upper_channel[i]) or 
            np.isnan(lower_channel[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_1w_aligned[i]
        upper_val = upper_channel[i]
        lower_val = lower_channel[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        # Determine EMA50 trend: rising if current > previous, falling if current < previous
        if i > 100:
            ema_prev = ema_1w_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long: price breaks above upper Donchian AND EMA50 rising AND volume spike
            if (price > upper_val and ema_rising and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below lower Donchian AND EMA50 falling AND volume spike
            elif (price < lower_val and ema_falling and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below lower Donchian OR EMA50 flattening (not rising)
                if (price < lower_val or not ema_rising):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price breaks above upper Donchian OR EMA50 flattening (not falling)
                if (price > upper_val or not ema_falling):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_1wEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0