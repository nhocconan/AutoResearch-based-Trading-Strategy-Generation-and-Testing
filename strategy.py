#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h breakout strategy with 4h trend filter and volume confirmation
# Enter long when: price breaks above 4h Donchian upper (20), volume > 2x avg, 1d close > 1w EMA(50), during active session (08-20 UTC)
# Enter short when: price breaks below 4h Donchian lower (20), volume > 2x avg, 1d close < 1w EMA(50), during active session
# Exit when price returns to 4h Donchian middle or opposite breakout occurs
# Uses weekly trend to filter breakouts in weak trends, targeting 80-150 trades over 4 years

name = "1h_donchian_breakout_4htrend_1wfilter_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_4h, donchian_mid)
    
    # 1d close > 1w EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    close_1d = df_1d['close'].values
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for indicators to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if position == 1:  # long position
            # Exit: price < 4h Donchian middle OR opposite breakout (price < 4h Donchian lower)
            if close[i] < donchian_mid_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price > 4h Donchian middle OR opposite breakout (price > 4h Donchian upper)
            if close[i] > donchian_mid_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries: breakout + volume + trend filter + session
            if in_session and volume[i] > volume_threshold[i]:
                if close[i] > donchian_high_aligned[i] and close_1d[i] > ema_50_1w_aligned[i]:
                    # Bullish breakout above 4h resistance with bullish weekly trend
                    signals[i] = 0.20
                    position = 1
                elif close[i] < donchian_low_aligned[i] and close_1d[i] < ema_50_1w_aligned[i]:
                    # Bearish breakout below 4h support with bearish weekly trend
                    signals[i] = -0.20
                    position = -1
    
    return signals