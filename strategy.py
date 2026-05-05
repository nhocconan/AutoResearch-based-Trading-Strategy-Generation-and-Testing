#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout with 12h EMA trend filter and volume confirmation
# Long when: Price breaks above Donchian(20) high AND 12h EMA(50) trending up AND volume > 1.5x 20-period average volume
# Short when: Price breaks below Donchian(20) low AND 12h EMA(50) trending down AND volume > 1.5x 20-period average volume
# Exit when price returns to Donchian midpoint (mean reversion)
# Donchian breakout captures volatility expansion after consolidation
# 12h EMA filter ensures we trade with higher timeframe trend
# Volume confirmation reduces false breakouts
# Works in both bull and bear markets by trading breakouts in direction of higher timeframe trend
# Target: 100-200 total trades over 4 years (25-50/year) with discrete sizing 0.25

name = "4h_DonchianBreakout_12hEMA_Volume"
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
    
    # Get 12h data ONCE before loop for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:  # Need enough for EMA(50)
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Donchian Channel (20) on 4h
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # Calculate volume confirmation: current volume > 1.5x 20-period average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(avg_volume_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h EMA trend direction
        ema_trend_up = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1] if i > 0 else False
        ema_trend_down = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1] if i > 0 else False
        
        if position == 0:
            # Long: Break above Donchian high with uptrend and volume confirmation
            if close[i] > highest_20[i] and ema_trend_up and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low with downtrend and volume confirmation
            elif close[i] < lowest_20[i] and ema_trend_down and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: return to Donchian midpoint (mean reversion)
            if close[i] < donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: return to Donchian midpoint (mean reversion)
            if close[i] > donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals