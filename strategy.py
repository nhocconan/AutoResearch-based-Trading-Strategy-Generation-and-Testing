#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with 1-day ATR filter and volume confirmation
# Long when price breaks above 12h Donchian high AND 1d volume > 1.5x 20-period average AND price > 1d EMA50
# Short when price breaks below 12h Donchian low AND 1d volume > 1.5x 20-period average AND price < 1d EMA50
# Uses volume confirmation to filter false breakouts and EMA50 for trend alignment.
# Target: 50-150 total trades over 4 years (12-37/year) to stay within optimal range.

name = "12h_donchian20_1d_vol_ema_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian Channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # 1-day EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # 1-day volume average (20-period)
    volume_series = pd.Series(volume)
    vol_ma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if EMA50 or volume MA not ready
        if np.isnan(ema50[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits: reverse position or trend change
        if position == 1:  # long position
            # Exit: price breaks below 12h Donchian low OR price crosses below EMA50
            if (close[i] <= donchian_low[i] or 
                close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above 12h Donchian high OR price crosses above EMA50
            if (close[i] >= donchian_high[i] or 
                close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation and trend filter
            # Long: price breaks above 12h Donchian high AND volume > 1.5x average AND price > EMA50
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 12h Donchian low AND volume > 1.5x average AND price < EMA50
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
    
    return signals