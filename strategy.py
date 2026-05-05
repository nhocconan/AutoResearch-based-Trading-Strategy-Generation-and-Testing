#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume spike confirmation
# Long when: price breaks above Donchian upper (20-period high) AND 1w EMA50 rising AND volume > 1.5x 20-period average volume
# Short when: price breaks below Donchian lower (20-period low) AND 1w EMA50 falling AND volume > 1.5x 20-period average volume
# Exit when price crosses opposite Donchian band (upper for shorts, lower for longs) OR 1w EMA50 flips direction
# Uses 1d timeframe with 1w HTF for EMA50 trend filter (target: 30-100 total over 4 years)
# Donchian channels provide clear breakout levels with built-in volatility adjustment
# 1w EMA50 ensures we only trade with the primary trend, avoiding counter-trend whipsaws
# Volume spike confirms institutional participation in breakouts
# Discrete position sizing (0.25) minimizes fee churn while maintaining adequate exposure

name = "1d_Donchian20_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50 calculation
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(50)
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d Donchian channels (20-period)
    # Upper band: highest high over last 20 periods
    # Lower band: lowest low over last 20 periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate volume confirmation: volume > 1.5x 20-period average volume
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above upper band AND EMA50 rising AND volume spike
            if (close[i] > donchian_upper[i] and 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below lower band AND EMA50 falling AND volume spike
            elif (close[i] < donchian_lower[i] and 
                  ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below lower band OR EMA50 starts falling
            if (close[i] < donchian_lower[i] or 
                ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above upper band OR EMA50 starts rising
            if (close[i] > donchian_upper[i] or 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals