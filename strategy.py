#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w trend filter and volume confirmation
# Long when: Price breaks above 12h Donchian upper (20) AND 1w close > 1w EMA50 AND 12h volume > 1.5x 20-period average
# Short when: Price breaks below 12h Donchian lower (20) AND 1w close < 1w EMA50 AND 12h volume > 1.5x 20-period average
# Exit when price touches opposite Donchian level or 1w EMA50 crossover
# Donchian provides clear breakout levels, 1w EMA50 filters trend direction, volume confirms institutional participation
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "12h_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:  # Need enough for EMA50
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian channels (20-period)
    # Use rolling window on 12h data directly
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 12h volume spike (current volume > 1.5x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check conditions
        vol_cond = bool(vol_spike[i])
        above_ema = close[i] > ema_50_1w_aligned[i]
        below_ema = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: Break above Donchian upper with 1w uptrend and volume spike
            if close[i] > donchian_upper[i] and above_ema and vol_cond:
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian lower with 1w downtrend and volume spike
            elif close[i] < donchian_lower[i] and below_ema and vol_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price touches Donchian lower or 1w EMA50 crossover down
            if close[i] <= donchian_lower[i] or (close[i] < ema_50_1w_aligned[i] and ema_50_1w_aligned[i-1] <= close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price touches Donchian upper or 1w EMA50 crossover up
            if close[i] >= donchian_upper[i] or (close[i] > ema_50_1w_aligned[i] and ema_50_1w_aligned[i-1] >= close[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals