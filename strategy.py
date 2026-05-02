#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 12h EMA50 trend filter + volume confirmation
# Donchian breakout provides clear entry/exit levels with proven edge in trending markets
# 12h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend trades
# Volume confirmation (>1.3x 20-period EMA) filters for institutional participation
# Designed for 4h timeframe targeting 20-50 trades/year (75-200 total over 4 years)
# Works in bull markets (price > Donchian upper + above 12h EMA50) and bear markets (price < Donchian lower + below 12h EMA50)
# Uses discrete position sizing (0.30) to balance return potential with drawdown control

name = "4h_Donchian20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA50 trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Donchian channels (20-period)
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_upper = rolling_max(high, 20)
    donchian_lower = rolling_min(low, 20)
    
    # Volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_confirmation = volume > (1.3 * vol_ema_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(volume_confirmation[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend bias from 12h EMA50
        bullish_bias = close[i] > ema_50_12h_aligned[i]
        bearish_bias = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            if close[i] > donchian_upper[i] and bullish_bias and volume_confirmation[i]:
                # Long: Price breaks above Donchian upper + above 12h EMA50 + volume confirmation
                signals[i] = 0.30
                position = 1
            elif close[i] < donchian_lower[i] and bearish_bias and volume_confirmation[i]:
                # Short: Price breaks below Donchian lower + below 12h EMA50 + volume confirmation
                signals[i] = -0.30
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below Donchian lower OR trend turns bearish
            if close[i] < donchian_lower[i] or not bullish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:  # Short position
            # Exit: Price breaks above Donchian upper OR trend turns bullish
            if close[i] > donchian_upper[i] or not bearish_bias:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals