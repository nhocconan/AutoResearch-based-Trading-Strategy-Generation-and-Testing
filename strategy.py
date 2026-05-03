#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian(20) breakout for direction and 1h RSI(14) for entry timing.
# 4h Donchian provides robust structure for breakouts in both bull and bear markets.
# 1h RSI(14) < 30 for longs and > 70 for shorts ensures we enter on pullbacks within the trend.
# Volume confirmation (current volume > 1.5x 20-period EMA) filters false breakouts.
# Session filter (08-20 UTC) reduces noise trades.
# Designed for 60-150 total trades over 4 years (15-37/year) with discrete position sizing (0.20).
# Works in bull markets via upward breaks at upper channel on RSI pullbacks and in bear markets via downward breaks at lower channel on RSI pullbacks.

name = "1h_Donchian20_RSI14_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Donchian(20) direction filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian channels (20-period)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    high_series_4h = pd.Series(high_4h)
    low_series_4h = pd.Series(low_4h)
    donchian_upper_4h = high_series_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_series_4h.rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian to 1h timeframe (waits for completed 4h bar)
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # Calculate 1h RSI(14) for entry timing
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume confirmation: 20-period EMA on 1h
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start from 14 to have valid RSI
        # Skip if any value is NaN or outside session
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or 
            np.isnan(rsi_values[i]) or np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long: price near lower Donchian (pullback) in uptrend alignment with RSI oversold and volume spike
            if close[i] > donchian_lower_4h_aligned[i] and close[i] < donchian_upper_4h_aligned[i] and \
               rsi_values[i] < 30 and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price near upper Donchian (pullback) in downtrend alignment with RSI overbought and volume spike
            elif close[i] < donchian_upper_4h_aligned[i] and close[i] > donchian_lower_4h_aligned[i] and \
                 rsi_values[i] > 70 and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: RSI overbought or price breaks above upper Donchian (taking profit)
            if rsi_values[i] > 70 or close[i] >= donchian_upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: RSI oversold or price breaks below lower Donchian (taking profit)
            if rsi_values[i] < 30 or close[i] <= donchian_lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals