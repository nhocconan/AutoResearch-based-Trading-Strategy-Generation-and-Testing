# 6H_Donchian_Breakout_WeeklyTrend_Volume
# Hypothesis: Donchian(20) breakout on 6h chart with weekly trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian high + weekly uptrend + volume > 1.5x average.
# Short when price breaks below 20-period Donchian low + weekly downtrend + volume > 1.5x average.
# Exit when price closes back inside the Donchian channel.
# Uses weekly trend to filter direction, reducing false signals in choppy markets.
# Target: 15-30 trades/year per symbol. Works in bull/bear by following weekly trend.

name = "6H_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "6h"
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
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Weekly trend (EMA50 on weekly data)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema50_1w
    trend_1w_down = close_1w < ema50_1w
    
    # Align weekly trend to 6h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i]) or
            np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        trend_up = trend_1w_up_aligned[i] > 0.5
        trend_down = trend_1w_down_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: breakout above Donchian high + weekly uptrend + volume
            if close[i] > donchian_high[i] and trend_up and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: breakout below Donchian low + weekly downtrend + volume
            elif close[i] < donchian_low[i] and trend_down and volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit when price closes back inside Donchian channel
            if close[i] < donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit when price closes back inside Donchian channel
            if close[i] > donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals