#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above 20-day high AND close > 1w EMA50 (uptrend) AND volume spike
# Short when price breaks below 20-day low AND close < 1w EMA50 (downtrend) AND volume spike
# Donchian channels provide clear structure with fewer whipsaws in ranging markets
# 1w EMA50 offers smooth multi-week trend filter to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Designed for low trade frequency (target: 30-100 over 4 years) to minimize fee drag
# Works in bull markets (trend-following breakouts) and bear markets (mean reversion at extremes with volume)

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
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels on 1d from previous 20 completed daily bars
    if len(high) >= 20:
        # Rolling high/low of last 20 completed daily bars (shifted by 1 to avoid look-ahead)
        highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
        lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    else:
        highest_20 = np.full(n, np.nan)
        lowest_20 = np.full(n, np.nan)
    
    # Volume confirmation on 1d with moderate threshold
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high AND uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > highest_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low AND downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < lowest_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 20-day high OR closes below 1w EMA50
            if close[i] < highest_20[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 20-day low OR closes above 1w EMA50
            if close[i] > lowest_20[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals