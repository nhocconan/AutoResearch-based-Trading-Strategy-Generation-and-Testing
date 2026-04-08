#!/usr/bin/env python3
# 1d_weekly_price_action_v1
# Hypothesis: Uses daily price action with weekly trend context and volume confirmation.
# Long when: Daily close > weekly EMA20, daily close > daily open (bullish candle), volume > 1.5x 20-day average.
# Short when: Daily close < weekly EMA20, daily close < daily open (bearish candle), volume > 1.5x 20-day average.
# Exit when price crosses weekly EMA20 in opposite direction or volume drops below average.
# Uses 3 conditions max to avoid overtrading. Target: 10-25 trades/year on daily timeframe.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_price_action_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    open_price = prices['open'].values
    volume = prices['volume'].values
    
    # Daily bullish/bearish candle
    bullish_candle = close > open_price
    bearish_candle = close < open_price
    
    # Weekly EMA20 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below weekly EMA20 or volume drops below average
            if close[i] < ema20_1w_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above weekly EMA20 or volume drops below average
            if close[i] > ema20_1w_aligned[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above weekly EMA20, bullish candle, volume surge
            if (close[i] > ema20_1w_aligned[i] and 
                bullish_candle[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below weekly EMA20, bearish candle, volume surge
            elif (close[i] < ema20_1w_aligned[i] and 
                  bearish_candle[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals