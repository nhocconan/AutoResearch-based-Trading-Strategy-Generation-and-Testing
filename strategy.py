#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band AND 1w EMA50 rising AND volume > 2.0x 20-period MA.
Short when price breaks below Donchian lower band AND 1w EMA50 falling AND volume > 2.0x 20-period MA.
Exit when price touches opposite Donchian band or 1w EMA50 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades on daily chart.
Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
Donchian provides clear structure, 1w EMA50 filters major trend, volume confirms breakout strength.
Designed to work in both bull and bear markets by following higher timeframe trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    donchian_upper = np.full(n, np.nan)
    donchian_lower = np.full(n, np.nan)
    
    for i in range(20, n):
        # Use lookback of 20 periods (excluding current bar to avoid look-ahead)
        donchian_upper[i] = np.max(high[i-20:i])
        donchian_lower[i] = np.min(low[i-20:i])
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian (needs 20), EMA50, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        ema_val = ema_50_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_aligned[i-1]
            ema_rising = ema_val > ema_prev
            ema_falling = ema_val < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 1d volume > 2.0x 20-period MA (strict to reduce trades)
        vol_filter = volume[i] > 2.0 * vol_ma_val
        
        if position == 0:
            # Long: Break above Donchian upper AND EMA50 rising AND volume filter
            if price > upper and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower AND EMA50 falling AND volume filter
            elif price < lower and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian lower (opposite) OR EMA50 starts falling
                if price < lower or (i >= start_idx + 1 and ema_val < ema_50_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian upper (opposite) OR EMA50 starts rising
                if price > upper or (i >= start_idx + 1 and ema_val > ema_50_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "1d"
leverage = 1.0