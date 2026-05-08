#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with weekly EMA trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high, weekly EMA(10) up, and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, weekly EMA(10) down, and volume > 1.5x average.
# Exit when price crosses Donchian midline (10-day average of high/low).
# Uses discrete position sizing (0.25) to minimize churn. Works in trending markets.
# Weekly EMA filter ensures alignment with higher timeframe trend, reducing counter-trend trades.

name = "1d_Donchian_WeeklyEMA_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # Weekly EMA(10) for trend filter
    close_1w_series = pd.Series(close_1w)
    ema_10_1w = close_1w_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    ema_10_up = ema_10_1w[1:] > ema_10_1w[:-1]
    ema_10_up = np.concatenate([[False], ema_10_up])
    
    # Align weekly EMA trend to daily
    ema_10_up_aligned = align_htf_to_ltf(prices, df_1w, ema_10_up.astype(float))
    
    # Volume average (20-period) for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_10_up_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend alignment
            if volume[i] > 1.5 * vol_avg_20[i]:
                # Long breakout: price above Donchian high, weekly EMA up
                if close[i] > high_20[i] and ema_10_up_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low, weekly EMA down
                elif close[i] < low_20[i] and not ema_10_up_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long position: exit when price crosses Donchian midline
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price crosses Donchian midline
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals