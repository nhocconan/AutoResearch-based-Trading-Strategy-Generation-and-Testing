#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band AND 12h EMA50 rising AND volume > 1.5x 20-bar average.
# Short when price breaks below Donchian lower band AND 12h EMA50 falling AND volume > 1.5x 20-bar average.
# Uses discrete sizing 0.25 to manage drawdown. Target: 75-200 total trades over 4 years (19-50/year).
# Donchian provides clear structure, EMA50 filters trend alignment, volume confirms breakout strength.
# Primary timeframe: 4h, HTF: 12h for EMA trend filter.

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Donchian(20) channels
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 trend filter (rising/falling)
    prev_close = df_12h['close'].values
    ema_50 = pd.Series(prev_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    # Trend: rising if current EMA > previous EMA, falling if current EMA < previous EMA
    ema_50_prev = np.roll(ema_50_aligned, 1)
    ema_50_prev[0] = ema_50_aligned[0]
    ema_rising = ema_50_aligned > ema_50_prev
    ema_falling = ema_50_aligned < ema_50_prev
    
    # Volume confirmation: current 4h volume > 1.5x 20-bar average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_vol = volume[i]
        curr_vol_ma = vol_ma[i]
        
        if curr_vol_ma <= 0:
            signals[i] = 0.0
            continue
            
        volume_confirm = curr_vol > (curr_vol_ma * 1.5)  # Volume confirmation threshold
        
        # Donchian breakout conditions
        breakout_up = curr_close > highest_high[i-1]  # Close above previous upper band
        breakout_down = curr_close < lowest_low[i-1]   # Close below previous lower band
        
        # Entry conditions
        if position == 0:  # Flat - look for new entries
            # Long: breakout up AND rising EMA50 trend AND volume confirmation
            if (breakout_up and 
                ema_rising[i] and 
                volume_confirm):
                signals[i] = 0.25
                position = 1
            # Short: breakout down AND falling EMA50 trend AND volume confirmation
            elif (breakout_down and 
                  ema_falling[i] and 
                  volume_confirm):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price retouches Donchian middle OR EMA trend turns falling
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if (curr_close <= donchian_mid or 
                ema_falling[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price retouches Donchian middle OR EMA trend turns rising
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if (curr_close >= donchian_mid or 
                ema_rising[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals