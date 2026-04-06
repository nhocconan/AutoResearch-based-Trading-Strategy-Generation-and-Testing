#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Index (Bull/Bear Power) with 12-hour trend filter and volume confirmation
# Elder Ray = Close - EMA13 (Bull Power), EMA13 - Close (Bear Power)
# Trades only when: 1) Bull Power > 0 and Bear Power < 0 (bullish alignment) OR vice versa for bearish
# 2) 12-hour EMA20 trend agrees with signal direction
# 3) Volume > 1.5x 20-period average for institutional confirmation
# Designed for 6h timeframe targeting 75-200 trades over 4 years (~19-50/year)

name = "6h_elder_ray12h_trend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate EMA13 for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = close - ema13  # Close - EMA13
    bear_power = ema13 - close  # EMA13 - Close
    
    # 12-hour EMA20 for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema20_12h)
    
    # 20-period volume average for confirmation
    vol_s = pd.Series(volume)
    vol_ma20 = vol_s.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period (need EMA13 and EMA20)
    start = max(13, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema20_12h_aligned[i]) or np.isnan(vol_ma20[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma20[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power becomes positive (bullish momentum fading) or stoploss
            if (bear_power[i] > 0 or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power becomes negative (bearish momentum fading) or stoploss
            if (bull_power[i] < 0 or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: Bull Power positive AND Bear Power negative (bullish alignment) AND price above 12h EMA20
                if (bull_power[i] > 0 and bear_power[i] < 0 and 
                    close[i] > ema20_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bull Power negative AND Bear Power positive (bearish alignment) AND price below 12h EMA20
                elif (bull_power[i] < 0 and bear_power[i] > 0 and 
                      close[i] < ema20_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals