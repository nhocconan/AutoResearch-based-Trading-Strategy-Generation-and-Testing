#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Fractal breakout with 1-day trend filter and volume confirmation.
# Long when: Bullish fractal breakout (price > recent high) AND 1-day EMA34 rising AND volume > 1.8 * EMA20(volume).
# Short when: Bearish fractal breakdown (price < recent low) AND 1-day EMA34 falling AND volume > 1.8 * EMA20(volume).
# Exit when price crosses back below/above the 6-period EMA.
# Williams Fractals identify swing points; breakouts from these levels with trend and volume confirmation
# capture momentum shifts. Works in bull markets via upward breakouts from support and in bear markets
# via downward breakouts from resistance. Target: 50-150 total trades over 4 years.
name = "6h_WilliamsFractal_1dEMA34_Volume"
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
    
    # Williams Fractals: 5-bar pattern (requires 2 bars on each side)
    # Bearish fractal: high[n-2] is highest of [n-4:n+1]
    # Bullish fractal: low[n-2] is lowest of [n-4:n+1]
    n_f = 5
    half = n_f // 2  # 2
    
    bearish_fractal = np.zeros(n, dtype=bool)
    bullish_fractal = np.zeros(n, dtype=bool)
    
    for i in range(half, n - half):
        # Bearish fractal: current high is highest in window
        if high[i] == np.max(high[i - half:i + half + 1]):
            bearish_fractal[i] = True
        # Bullish fractal: current low is lowest in window
        if low[i] == np.min(low[i - half:i + half + 1]):
            bullish_fractal[i] = True
    
    # EMA6 for exit
    ema_6 = pd.Series(close).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    # Load 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # EMA34 on 1d close
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    # Rising if current > previous, falling if current < previous
    ema_34_rising = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_falling = np.zeros_like(ema_34_1d, dtype=bool)
    ema_34_rising[1:] = ema_34_1d[1:] > ema_34_1d[:-1]
    ema_34_falling[1:] = ema_34_1d[1:] < ema_34_1d[:-1]
    
    # Williams Fractals need 2 extra 1d bars for confirmation (center bar + 2 following)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_1d, ema_34_rising, additional_delay_bars=2)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_1d, ema_34_falling, additional_delay_bars=2)
    
    # Volume confirmation: current volume > 1.8 * 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(ema_6[i]) or np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i]) or 
            np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bullish fractal breakout (price > recent high) AND EMA34(1d) rising AND volume spike
            long_condition = bullish_fractal[i] and (close[i] > high[i-2]) and ema_34_rising_aligned[i] and volume_spike[i]
            # Short: Bearish fractal breakdown (price < recent low) AND EMA34(1d) falling AND volume spike
            short_condition = bearish_fractal[i] and (close[i] < low[i-2]) and ema_34_falling_aligned[i] and volume_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close < EMA6
            if close[i] < ema_6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close > EMA6
            if close[i] > ema_6[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals