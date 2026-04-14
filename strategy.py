#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Bollinger Band Reversion with 1d Trend Filter
# In 6b channels, price tends to revert to the mean after touching Bollinger Bands
# Trend filter from 1d EMA200 prevents counter-trend trades
# Works in bull/bear by only taking mean-reversion trades in the direction of higher timeframe trend
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h Bollinger Bands (20, 2.0)
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2.0 * bb_std
    bb_lower = bb_middle - 2.0 * bb_std
    
    # 1d EMA200 for trend filter (calculated once)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    ema_200 = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 50  # for Bollinger Bands and volume average
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Determine trend from 1d EMA200
        uptrend = price > ema_200_aligned[i]
        downtrend = price < ema_200_aligned[i]
        
        if position == 0:
            # Long: price touches lower BB in uptrend with volume confirmation
            if price <= bb_lower[i] and uptrend and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: price touches upper BB in downtrend with volume confirmation
            elif price >= bb_upper[i] and downtrend and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to middle band or reverses
            if price >= bb_middle[i] or price < bb_lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price returns to middle band or reverses
            if price <= bb_middle[i] or price > bb_upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_Bollinger_Reversion_1dEMA200_Filter"
timeframe = "6h"
leverage = 1.0