#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and volume confirmation
# - Uses 1w EMA(34) for primary trend direction (long above EMA, short below)
# - Enters on 1d Donchian(20) breakout in direction of 1w trend
# - Requires 1d volume > 1.5x 20-period average for confirmation
# - Exits on opposite Donchian(10) breakout or when price crosses 1w EMA
# - Position sizing: 0.25 (25% of capital) to balance reward and drawdown
# - Target: 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to avoid fee drag
# - Works in bull markets via trend-following breakouts, in bear markets via short breakdowns

name = "1d_1w_donchian_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # Pre-compute 1w EMA(34) for trend filter
    close_1w = df_1w['close'].values
    ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # 1d Donchian channels
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) for entries
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exits
    highest_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    lowest_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_1w_aligned[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(highest_10[i]) or np.isnan(lowest_10[i]) or
            np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit conditions: Donchian(10) breakdown or price crosses below 1w EMA
            if close[i] <= lowest_10[i] or close[i] < ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: Donchian(10) breakout or price crosses above 1w EMA
            if close[i] >= highest_10[i] or close[i] > ema_1w_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian(20) breakout with volume confirmation
            if (close[i] > highest_20[i] and  # Bullish breakout
                close[i] > ema_1w_aligned[i] and  # Above 1w EMA (uptrend)
                volume[i] > 1.5 * vol_ma_20[i]):  # Volume confirmation
                position = 1
                signals[i] = 0.25
            elif (close[i] < lowest_20[i] and  # Bearish breakdown
                  close[i] < ema_1w_aligned[i] and  # Below 1w EMA (downtrend)
                  volume[i] > 1.5 * vol_ma_20[i]):  # Volume confirmation
                position = -1
                signals[i] = -0.25
    
    return signals