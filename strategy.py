# 12h_camarilla_breakout_v1
# Hypothesis: 12h Camarilla breakout with daily trend filter and volume confirmation.
# - Use 12h timeframe for fewer trades, lower fee drag
# - Daily trend filter (close above/below daily pivot) to avoid counter-trend trades
# - Volume confirmation (current volume > 1.5x 20-period average) to filter weak breakouts
# - Enter long when 12h close breaks above H3 resistance AND daily trend bullish AND volume spike
# - Enter short when 12h close breaks below L3 support AND daily trend bearish AND volume spike
# - Exit when price returns to daily pivot (mean reversion to value area)
# - Designed for 5-15 trades/year to avoid overtrading, works in bull/bear via trend filter

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data (same as input)
    df_12h = prices.copy()
    
    # Previous 12h OHLC for Camarilla calculation
    high_12h_prev = df_12h['high'].shift(1).values
    low_12h_prev = df_12h['low'].shift(1).values
    close_12h_prev = df_12h['close'].shift(1).values
    
    # Camarilla levels for 12h
    # H3 = close + 1.1*(high-low)/2
    # L3 = close - 1.1*(high-low)/2
    # Using previous period for calculation
    camarilla_range = high_12h_prev - low_12h_prev
    h3 = close_12h_prev + 1.1 * camarilla_range / 2
    l3 = close_12h_prev - 1.1 * camarilla_range / 2
    
    # Daily trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Daily pivot point
    daily_pivot = (high_1d + low_1d + close_1d) / 3
    # Align daily pivot to 12h
    daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    # Daily trend: bullish if close > pivot, bearish if close < pivot
    daily_bullish = close_1d > daily_pivot
    daily_bearish = close_1d < daily_pivot
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish.astype(float))
    daily_bearish_aligned = align_htf_to_ltf(prices, df_1d, daily_bearish.astype(float))
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(daily_pivot_aligned[i]) or 
            np.isnan(daily_bullish_aligned[i]) or 
            np.isnan(daily_bearish_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price returns to daily pivot (mean reversion)
            if close[i] <= daily_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price returns to daily pivot (mean reversion)
            if close[i] >= daily_pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long: 12h close breaks above H3 + daily bullish + volume spike
            if (close[i] > h3[i] and 
                daily_bullish_aligned[i] > 0.5 and 
                volume_spike[i]):
                position = 1
                signals[i] = 0.25
            # Short: 12h close breaks below L3 + daily bearish + volume spike
            elif (close[i] < l3[i] and 
                  daily_bearish_aligned[i] > 0.5 and 
                  volume_spike[i]):
                position = -1
                signals[i] = -0.25
    
    return signals