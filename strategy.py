#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA34 trend filter and volume confirmation
# Donchian channels provide robust price structure for breakouts in both bull and bear markets.
# 1w EMA34 ensures alignment with long-term trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts.
# Designed for 7-25 trades/year on 1d to minimize fee drag while capturing strong trending moves.
# Works in bull markets via long upper breakouts in uptrend and bear markets via short lower breakouts in downtrend.

name = "1d_Donchian20_EMA34_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) channels on 1d
    # Upper channel = highest high over last 20 periods
    # Lower channel = lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (no additional delay needed for price channels)
    # Since we're calculating on 1d data and aligning to 1d, it's 1:1
    # But we still use the helper for consistency and proper handling of data gaps
    donchian_upper_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high, 'low': low, 'close': close}), donchian_lower)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian AND 1w uptrend AND volume spike
            if (close[i] > donchian_upper_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and  # 1w uptrend
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below lower Donchian AND 1w downtrend AND volume spike
            elif (close[i] < donchian_lower_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and  # 1w downtrend
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below midpoint of Donchian channel OR 1w trend turns down
            donchian_mid = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2
            if close[i] < donchian_mid or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above midpoint of Donchian channel OR 1w trend turns up
            donchian_mid = (donchian_upper_aligned[i] + donchian_lower_aligned[i]) / 2
            if close[i] > donchian_mid or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals