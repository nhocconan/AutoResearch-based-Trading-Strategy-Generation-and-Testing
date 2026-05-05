#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA50 (strong uptrend) AND volume spike
# Short when price breaks below 20-day low AND price < 1w EMA50 (strong downtrend) AND volume spike
# Uses 1d timeframe as primary, with 1w EMA50 for trend filter to reduce whipsaw
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)

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
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d (20-period high/low)
    if len(high) >= 20:
        # Rolling max/min for high/low
        high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
        low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Shift by 1 to use only completed daily bar (look-ahead safety)
        donchian_high = np.roll(high_roll_max, 1)
        donchian_low = np.roll(low_roll_min, 1)
    else:
        donchian_high = np.full(n, np.nan)
        donchian_low = np.full(n, np.nan)
    
    # Volume confirmation on 1d (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to roll or insufficient data)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high AND strong uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND strong downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-day high OR closes below 1w EMA50
            if close[i] < donchian_high[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-day low OR closes above 1w EMA50
            if close[i] > donchian_low[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals