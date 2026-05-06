#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# Long when price breaks above 20-day high AND price > 1w EMA200 AND volume > 2.0 * 20-day avg volume
# Short when price breaks below 20-day low AND price < 1w EMA200 AND volume > 2.0 * 20-day avg volume
# Exit when price crosses 10-day EMA (mean reversion to short-term trend)
# Uses discrete sizing 0.25 to control drawdown in bear markets
# Target: 50-80 total trades over 4 years (12-20/year) for 1d timeframe
# Donchian channels provide structural breakouts, 1w EMA200 filters primary trend, volume confirms conviction

name = "1d_Donchian20_1wEMA200_Trend_VolumeSpike_v1"
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
    
    # Get daily data ONCE before loop for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels (using prior 20 periods, not including current)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 10-period EMA for exit
    close_1d_series = pd.Series(close_1d)
    ema_10_1d = close_1d_series.ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Get weekly data ONCE before loop for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200
    close_1w_series = pd.Series(close_1w)
    ema_200_1w = close_1w_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-day average volume
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    # Align HTF indicators to 1d timeframe (wait for completed HTF bar)
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_10_1d_aligned[i]) or np.isnan(ema_200_1w_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above 20-day high with uptrend and volume spike
            if (close[i] > high_20_aligned[i] and close[i] > ema_200_1w_aligned[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-day low with downtrend and volume spike
            elif (close[i] < low_20_aligned[i] and close[i] < ema_200_1w_aligned[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below 10-day EMA (mean reversion)
            if close[i] < ema_10_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above 10-day EMA (mean reversion)
            if close[i] > ema_10_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals