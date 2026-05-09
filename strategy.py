#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA50 trend filter and volume spike confirmation
# Designed for low trade frequency to minimize fee drag (target: 50-150 trades over 4 years).
# Uses Donchian channel breakouts for trend continuation, filtered by 1d EMA50 direction and volume spikes.
# Works in both bull and bear markets: long when price breaks above upper band in uptrend,
# short when price breaks below lower band in downtrend.
name = "12h_Donchian20_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on 1d close
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channel (20-period) on 12h data
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Need 50 for EMA50 and 20 for Donchian
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1d = ema_50_1d_aligned[i]
        upper_band = donchian_upper[i]
        lower_band = donchian_lower[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for spike detection
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: Close > Upper Donchian band AND price > 1d EMA50 (uptrend) AND volume > 2x average
            if close[i] > upper_band and close[i] > ema_1d and vol > 2.0 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Close < Lower Donchian band AND price < 1d EMA50 (downtrend) AND volume > 2x average
            elif close[i] < lower_band and close[i] < ema_1d and vol > 2.0 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Close < Lower Donchian band OR trend reverses (price < 1d EMA50)
            if close[i] < lower_band or close[i] < ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Close > Upper Donchian band OR trend reverses (price > 1d EMA50)
            if close[i] > upper_band or close[i] > ema_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals