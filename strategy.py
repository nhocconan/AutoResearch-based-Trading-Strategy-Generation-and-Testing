#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian breakout with 1d trend filter and volume confirmation
# Long when: Price breaks above Donchian(20) high AND 1d EMA(34) rising AND volume > 1.5x average
# Short when: Price breaks below Donchian(20) low AND 1d EMA(34) falling AND volume > 1.5x average
# Exit when price returns to Donchian(20) midline
# Designed for 6h timeframe with moderate trade frequency (target: 15-35/year) to avoid fee drag.
# Uses Donchian for breakout signals, 1d EMA for trend filter, and volume for confirmation.
# Works in bull markets via upside breakouts in uptrend, in bear markets via downside breakouts in downtrend.
# Volume filter avoids false breakouts in low-volume periods.

name = "6h_Donchian_1dEMA34_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # 1d EMA(34) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):  # 20-period MA
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: breakout above Donchian high + 1d EMA rising + volume confirmation
            long_condition = (close[i] > donchian_high[i]) and ema_34_rising_aligned[i] and volume_confirm[i]
            # Short: breakdown below Donchian low + 1d EMA falling + volume confirmation
            short_condition = (close[i] < donchian_low[i]) and ema_34_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midline
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midline
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals