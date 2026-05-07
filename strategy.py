#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high AND 1w EMA(50) rising AND volume > 20-day average volume
# Short when: price breaks below Donchian(20) low AND 1w EMA(50) falling AND volume > 20-day average volume
# Exit when price crosses back through the Donchian(20) midline (mean of high and low).
# Designed for 1d timeframe with low trade frequency (target: 15-25/year) to avoid fee drag.
# Uses 1w for trend direction and volume confirmation to avoid false breakouts.
# Works in bull markets via breakouts in uptrend, in bear markets via breakdowns in downtrend.
# Volume filter avoids low-conviction breakouts that often fail.
name = "1d_Donchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_1w, dtype=bool)
    ema_50_rising[1:] = ema_50_1w[1:] > ema_50_1w[:-1]
    ema_50_falling[1:] = ema_50_1w[1:] < ema_50_1w[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_50_falling)
    
    # Volume confirmation: current volume > 20-day average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND 1w EMA50 rising AND volume confirmed
            long_condition = (close[i] > donchian_high[i]) and ema_50_rising_aligned[i] and volume_confirmed[i]
            # Short: break below Donchian low AND 1w EMA50 falling AND volume confirmed
            short_condition = (close[i] < donchian_low[i]) and ema_50_falling_aligned[i] and volume_confirmed[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Donchian mid
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Donchian mid
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals