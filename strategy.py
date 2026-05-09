#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA(20) trend filter and volume confirmation
# Long when: close > Donchian upper (20-day high), 1w EMA(20) rising, volume spike (>1.5x 20-day avg)
# Short when: close < Donchian lower (20-day low), 1w EMA(20) falling, volume spike
# Exit when: price crosses the Donchian midline (10-day average of high/low) OR trend reverses
# Position size: 0.25 to limit drawdown. Target: 15-25 trades/year to avoid fee drag.
# Designed to work in bull (breakouts) and bear (mean-reversion at extremes) via trend filter.

name = "1d_Donchian_20_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian channels (20-period) on daily data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2  # Midline for exit
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 1w EMA(20) for trend filter
    close_1w = df_1w['close']
    ema_20_1w = close_1w.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_prev = np.roll(ema_20_1w, 1)
    ema_20_1w_prev[0] = ema_20_1w[0]
    ema_rising = ema_20_1w > ema_20_1w_prev
    ema_falling = ema_20_1w < ema_20_1w_prev
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Volume spike: current volume > 1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(donchian_mid[i]) or np.isnan(ema_rising_aligned[i]) or
            np.isnan(ema_falling_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > Donchian high + 1w EMA rising + volume spike
            if (close[i] > donchian_high[i] and 
                ema_rising_aligned[i] and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price < Donchian low + 1w EMA falling + volume spike
            elif (close[i] < donchian_low[i] and 
                  ema_falling_aligned[i] and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below Donchian midline OR trend turns down
            if (close[i] < donchian_mid[i]) or (not ema_rising_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above Donchian midline OR trend turns up
            if (close[i] > donchian_mid[i]) or (not ema_falling_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals