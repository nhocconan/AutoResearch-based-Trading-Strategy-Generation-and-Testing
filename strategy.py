# 4H_Donchian_Breakout_With_1dTrend_And_Volume
# Hypothesis: Donchian channel breakout with 1d trend filter and volume confirmation.
# Long when: price breaks above Donchian(20) high + 1d uptrend + volume > 2x average.
# Short when: price breaks below Donchian(20) low + 1d downtrend + volume > 2x average.
# Exit when: price closes back inside the Donchian channel.
# Uses 1d trend for higher timeframe context to avoid counter-trend trades.
# Volume filter reduces false breakouts. Designed for 4h timeframe to target 20-50 trades/year.
# Works in bull markets by following uptrend, in bear markets by following downtrend.

name = "4H_Donchian_Breakout_With_1dTrend_And_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # 1d trend (EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema50_1d
    trend_1d_down = close_1d < ema50_1d
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 2.0
        
        trend_up = trend_1d_up_aligned[i] > 0.5
        trend_down = trend_1d_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: breakout above Donchian high + 1d uptrend + volume
            if close[i] > high_roll[i] and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below Donchian low + 1d downtrend + volume
            elif close[i] < low_roll[i] and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price closes back inside Donchian channel
            if close[i] < high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes back inside Donchian channel
            if close[i] > low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals