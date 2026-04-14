#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index + 1d EMA200 Trend Filter + Volume Confirmation
# Elder Ray measures bull/bear power relative to EMA13. In trending markets,
# we go long when bull power > 0 and price > 1d EMA200, short when bear power < 0 and price < 1d EMA200.
# Volume confirmation ensures breakouts have participation. Works in bull/bear by following the trend.
# Target: 50-150 total trades over 4 years (12-37/year)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Elder Ray components (13-period EMA for reference)
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema13
    bear_power = low - ema13
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Volume confirmation: volume > 1.5x average volume (20-period)
    vol_series = pd.Series(volume)
    avg_vol = vol_series.rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(50, 200)  # for EMA200
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema200_1d_aligned[i]) or np.isnan(avg_vol[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Trend filter: price must be on correct side of 1d EMA200
        if ema200_1d_aligned[i] <= 0:  # invalid EMA
            signals[i] = 0.0
            continue
            
        if position == 0:
            # Long: bull power positive AND price above 1d EMA200 with volume
            if bull_power[i] > 0 and price > ema200_1d_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = 1
                signals[i] = position_size
            # Short: bear power negative AND price below 1d EMA200 with volume
            elif bear_power[i] < 0 and price < ema200_1d_aligned[i] and vol > 1.5 * avg_vol[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: bear power turns negative OR price drops below EMA200
            if bear_power[i] < 0 or price < ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: bull power turns positive OR price rises above EMA200
            if bull_power[i] > 0 or price > ema200_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_ElderRay_1dEMA200_VolumeFilter"
timeframe = "6h"
leverage = 1.0