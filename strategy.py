#!/usr/bin/env python3
# Hypothesis: 12h Williams %R mean reversion with 1d EMA50 trend filter and volume spike confirmation.
# Long when Williams %R < -80 (oversold) AND price > 1d EMA50 AND volume > 1.5x average.
# Short when Williams %R > -20 (overbought) AND price < 1d EMA50 AND volume > 1.5x average.
# Exit when Williams %R returns to -50 (mean reversion) or trend reverses.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Williams %R is effective in ranging markets (common in 2025 BTC/ETH bear/range) and captures mean reversion.
# 12h timeframe reduces trade frequency vs lower TFs, improving fee drag profile.

name = "12h_WilliamsR_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Williams %R calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Williams %R on 12h: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    # Using 14-period lookback
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    close_series = pd.Series(close_12h)
    highest_high = high_series.rolling(window=14, min_periods=14).max().values
    lowest_low = low_series.rolling(window=14, min_periods=14).min().values
    williams_r = ((highest_high - close_series) / (highest_high - lowest_low)) * -100
    williams_r = williams_r.values  # Convert to numpy array
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Williams %R oversold (< -80) AND price > 1d EMA50 AND volume spike
            if williams_r[i] < -80 and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Williams %R overbought (> -20) AND price < 1d EMA50 AND volume spike
            elif williams_r[i] > -20 and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Williams %R returns to -50 (mean reversion) OR trend reversal (price < 1d EMA50)
            if williams_r[i] > -50 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Williams %R returns to -50 (mean reversion) OR trend reversal (price > 1d EMA50)
            if williams_r[i] < -50 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals