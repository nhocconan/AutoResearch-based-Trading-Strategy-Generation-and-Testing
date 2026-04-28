#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakout with volume confirmation and 1d trend filter (EMA50).
# Enter long when price breaks above 4h Donchian upper channel with volume >1.5x 20-bar average and price > 1d EMA50.
# Enter short when price breaks below 4h Donchian lower channel with volume >1.5x 20-bar average and price < 1d EMA50.
# Uses session filter (08-20 UTC) to avoid low-liquidity periods.
# Position size: 0.20 (discrete) to control drawdown and fees.
# Target: 15-35 trades/year (60-140 over 4 years) to avoid fee drag.
# Donchian provides objective breakout levels, volume confirms breakout strength, 1d EMA ensures alignment with higher timeframe trend.
# Works in bull (breakouts with trend) and bear (failed breaks reverse via exits) markets.

name = "1h_Donchian20_4hVolume_1dEMA50_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian channel (HTF)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h Donchian channel (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    n_4h = len(high_4h)
    donchian_high = np.full(n_4h, np.nan)
    donchian_low = np.full(n_4h, np.nan)
    
    for i in range(20, n_4h):
        donchian_high[i] = np.max(high_4h[i-20:i])
        donchian_low[i] = np.min(low_4h[i-20:i])
    
    # Forward fill Donchian levels
    donchian_high = pd.Series(donchian_high).ffill().values
    donchian_low = pd.Series(donchian_low).ffill().values
    
    # Align 4h indicators to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA50 trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate 1h volume spike: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Donchian breakout conditions with volume confirmation and 1d trend filter
        long_breakout = close[i] > donchian_high_aligned[i] and volume_spike[i] and close[i] > ema_50_aligned[i]
        short_breakout = close[i] < donchian_low_aligned[i] and volume_spike[i] and close[i] < ema_50_aligned[i]
        
        # Exit conditions: opposite Donchian level
        long_exit = close[i] < donchian_low_aligned[i]
        short_exit = close[i] > donchian_high_aligned[i]
        
        # Handle entries and exits
        if long_breakout and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.20
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals