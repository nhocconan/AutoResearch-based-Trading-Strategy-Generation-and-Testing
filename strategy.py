# -*- coding: utf-8 -*-
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA trend filter and volume confirmation.
# Long when price breaks above Donchian high(20) AND 12h EMA(50) rising AND volume > 1.5x average volume.
# Short when price breaks below Donchian low(20) AND 12h EMA(50) falling AND volume > 1.5x average volume.
# Exit when price crosses back to Donchian midline or opposite breakout occurs.
# Uses volume confirmation to avoid false breakouts and EMA trend filter for direction.
# Designed for 4h timeframe with tight entry conditions (target: 25-40 trades/year).
# Works in bull markets via upward breakouts in uptrend, in bear markets via downward breakouts in downtrend.
name = "4h_DonchianBreakout_12hEMA50_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 12h EMA(50) for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_rising = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_falling = np.zeros_like(ema_50_12h, dtype=bool)
    ema_50_rising[1:] = ema_50_12h[1:] > ema_50_12h[:-1]
    ema_50_falling[1:] = ema_50_12h[1:] < ema_50_12h[:-1]
    
    ema_50_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_50_rising)
    ema_50_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_50_falling)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(ema_50_rising_aligned[i]) or np.isnan(ema_50_falling_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high AND 12h EMA50 rising AND volume confirmation
            long_condition = (close[i] > donch_high[i]) and ema_50_rising_aligned[i] and volume_confirm[i]
            # Short: break below Donchian low AND 12h EMA50 falling AND volume confirmation
            short_condition = (close[i] < donch_low[i]) and ema_50_falling_aligned[i] and volume_confirm[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: cross below Donchian midline OR opposite breakout with volume confirmation
            exit_condition = (close[i] < donch_mid[i]) or \
                            ((close[i] < donch_low[i]) and ema_50_falling_aligned[i] and volume_confirm[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: cross above Donchian midline OR opposite breakout with volume confirmation
            exit_condition = (close[i] > donch_mid[i]) or \
                            ((close[i] > donch_high[i]) and ema_50_rising_aligned[i] and volume_confirm[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals