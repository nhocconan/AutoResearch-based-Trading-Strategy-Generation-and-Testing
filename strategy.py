#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w trend filter + volume confirmation
# Donchian channel breakouts provide clear trend-following signals with defined risk.
# 1w EMA50 acts as a strong trend filter to avoid counter-trend trades.
# Volume confirmation (>1.5x 20-period EMA) ensures breakouts have conviction.
# Designed for 1d timeframe targeting 30-100 total trades over 4 years (7-25/year).
# Uses discrete position sizing (0.25) to minimize fee churn and manage drawdown.
# Works in both bull and bear markets by only taking trades in the direction of the 1w trend.

name = "1d_Donchian20_1wEMA50_Trend_Volume"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    # We need 20 periods of high/low to calculate the channel
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Pre-calculate volume confirmation: 20-period EMA of volume
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    for i in range(20, n):  # Start after we have enough data for Donchian
        # Skip if any value is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Donchian channels for the current bar (using past 20 bars, not including current)
        # Donchian upper = max(high[i-20:i])
        # Donchian lower = min(low[i-20:i])
        start_idx = i - 20
        end_idx = i  # exclusive
        
        if start_idx < 0:
            # Not enough data yet
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        donchian_upper = np.max(high[start_idx:end_idx])
        donchian_lower = np.min(low[start_idx:end_idx])
        
        # Volume confirmation: current volume > 1.5 x 20-period EMA
        volume_confirm = volume[i] > (1.5 * vol_ema_20[i])
        
        if position == 0:
            # Long breakout: price closes above Donchian upper + volume + 1w EMA50 uptrend
            if (close[i] > donchian_upper and 
                volume_confirm and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below Donchian lower + volume + 1w EMA50 downtrend
            elif (close[i] < donchian_lower and 
                  volume_confirm and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Donchian lower OR 1w EMA50 turns down
            if (close[i] < donchian_lower or 
                close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Donchian upper OR 1w EMA50 turns up
            if (close[i] > donchian_upper or 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals