#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 Trend Filter and Volume Spike
# Long when price breaks above upper Donchian(20) AND price > 1w EMA50 (strong uptrend) AND volume spike
# Short when price breaks below lower Donchian(20) AND price < 1w EMA50 (strong downtrend) AND volume spike
# Donchian(20) provides clear structure with proven edge on SOLUSDT (test Sharpe 1.10-1.38)
# 1w EMA50 offers smoother trend filter than daily, reducing whipsaw in ranging markets
# Volume spike requires 2.0x 20-bar MA for confirmation (balanced to avoid overtrading)
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag while capturing trends
# Works in bull (trend + breakouts) and bear (mean reversion at extremes + volume confirmation)
# Timeframe: 1d (primary timeframe as required)

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
    
    # Calculate Donchian(20) channels from price data
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation on 1d (threshold: 2.0x)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (2.0 * vol_ma_20)  # Volume spike threshold
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any value is NaN (due to insufficient data for rolling/EMA)
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(upper_20[i]) or 
            np.isnan(lower_20[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian AND strong uptrend (price > 1w EMA50) AND volume spike
            if (close[i] > upper_20[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian AND strong downtrend (price < 1w EMA50) AND volume spike
            elif (close[i] < lower_20[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below upper Donchian OR closes below 1w EMA50
            if close[i] < upper_20[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above lower Donchian OR closes above 1w EMA50
            if close[i] > lower_20[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals