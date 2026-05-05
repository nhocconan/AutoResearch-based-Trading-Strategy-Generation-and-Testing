#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian upper AND close > EMA50(1w) AND volume > 1.5x 20-period average
# Short when price breaks below Donchian lower AND close < EMA50(1w) AND volume > 1.5x 20-period average
# Exit when price retracement to Donchian midpoint OR EMA50(1w) trend flip
# Uses 4h primary timeframe with 1w HTF for trend filter to reduce whipsaw and avoid overtrading
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 100-200 total trades over 4 years (25-50/year) to balance opportunity and cost
# Donchian channels provide robust structure; breakouts with volume and weekly trend filter capture strong moves
# Proven pattern from DB: Donchian breakout + volume + trend filter works well on SOLUSDT (test Sharpe 1.10-1.38)

name = "4h_Donchian20_1wEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data ONCE before loop for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w close for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate Donchian channels from 4h OHLC (20-period)
    if len(high) >= 20:
        # Donchian upper: highest high over last 20 periods
        donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
        # Donchian lower: lowest low over last 20 periods
        donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
        # Donchian midpoint: (upper + lower) / 2
        donchian_mid = (donchian_upper + donchian_lower) / 2.0
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
        donchian_mid = np.full(n, np.nan)
    
    # Volume confirmation: volume > 1.5x 20-period average (balanced to avoid overtrading)
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.5 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper AND close > EMA50(1w) AND volume spike
            if (high[i] > donchian_upper[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian lower AND close < EMA50(1w) AND volume spike
            elif (low[i] < donchian_lower[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price retracement to Donchian midpoint OR close < EMA50(1w) (trend flip)
            if close[i] <= donchian_mid[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price retracement to Donchian midpoint OR close > EMA50(1w) (trend flip)
            if close[i] >= donchian_mid[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals