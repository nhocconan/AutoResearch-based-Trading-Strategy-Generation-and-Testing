#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout with volume confirmation and 1d EMA200 trend filter.
# Enter long when price breaks above 4h Donchian(20) high with volume spike and above 1d EMA200.
# Enter short when price breaks below 4h Donchian(20) low with volume spike and below 1d EMA200.
# Uses discrete position sizing (0.20) to minimize fee churn. Target: 60-150 total trades over 4 years.
# Donchian channels provide structure, volume confirms breakout strength, EMA200 filters trend direction.
# Works in bull (breakouts with trend) and bear (failed breaks reverse) markets.

name = "1h_Donchian20_4hVolSpike_1dEMA200_Trend_v1"
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    n_4h = len(high_4h)
    donchian_high = np.full(n_4h, np.nan)
    donchian_low = np.full(n_4h, np.nan)
    
    for i in range(n_4h):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high_4h[start_idx:i+1])
        donchian_low[i] = np.min(low_4h[start_idx:i+1])
    
    # Forward fill Donchian levels
    donchian_high = pd.Series(donchian_high).ffill().values
    donchian_low = pd.Series(donchian_low).ffill().values
    
    # Align 4h Donchian to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 200:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMA to 1h timeframe
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate 1h volume spike: >2.0x 24-bar average volume (6h equivalent)
    volume_series = pd.Series(volume)
    volume_ma_24 = volume_series.rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > 2.0 * volume_ma_24
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: price relative to 1d EMA200
        above_ema = close[i] > ema_200_1d_aligned[i]
        below_ema = close[i] < ema_200_1d_aligned[i]
        
        # Donchian breakout conditions with volume confirmation
        long_breakout = close[i] > donchian_high_aligned[i] and volume_spike[i]
        short_breakout = close[i] < donchian_low_aligned[i] and volume_spike[i]
        
        # Exit conditions: opposite Donchian level or trend reversal
        long_exit = close[i] < donchian_low_aligned[i] or below_ema
        short_exit = close[i] > donchian_high_aligned[i] or above_ema
        
        # Handle entries and exits
        if long_breakout and above_ema and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and below_ema and position >= 0:
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