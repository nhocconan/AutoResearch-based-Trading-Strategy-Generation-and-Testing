#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_donchian_breakout_volume
# Uses 4-hour Donchian channel breakout with volume confirmation and 12-hour trend filter.
# Long when price breaks above Donchian(20) high, volume > 1.5x 20-period average, and 12h EMA25 > EMA50.
# Short when price breaks below Donchian(20) low, volume > 1.5x 20-period average, and 12h EMA25 < EMA50.
# Designed for 4h timeframe to capture medium-term trends with low trade frequency (target: 20-50 trades/year).
# Works in bull markets (buying breakouts) and bear markets (selling breakdowns).

name = "4h_12h_donchian_breakout_volume"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate EMA25 and EMA50 on 12h close
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMAs to 4h timeframe (12h values update after 12h bar closes)
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channel on 4h (20-period high/low)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5 * 20-period average (4h timeframe)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema25_12h_aligned[i]) or np.isnan(ema50_12h_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            # Hold current position if volume filter fails
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long signal: price breaks above Donchian high, volume confirmed, and 12h EMA25 > EMA50
        if close[i] > donchian_high[i] and ema25_12h_aligned[i] > ema50_12h_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: price breaks below Donchian low, volume confirmed, and 12h EMA25 < EMA50
        elif close[i] < donchian_low[i] and ema25_12h_aligned[i] < ema50_12h_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: opposite Donchian breakout
        elif close[i] < donchian_high[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif close[i] > donchian_low[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals