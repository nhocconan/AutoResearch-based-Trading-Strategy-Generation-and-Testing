#!/usr/bin/env python3
"""
1h_AVWAP_Reversion_With_DailyTrend
Hypothesis: On 1h timeframe, enter long when price deviates below 1-period Volume-Weighted Average Price (VWAP) with volume confirmation during uptrend days, and short when price deviates above VWAP with volume confirmation during downtrend days. Uses daily trend filter (price vs 100-period EMA) to avoid counter-trend trades. VWAP mean reversion works in both bull/bear markets as it captures short-term exhaustion moves. Designed for low trade frequency (<30/year) by requiring volume surge and trend alignment.
"""

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
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 100:
        return np.zeros(n)
    
    # Calculate daily 100 EMA for trend filter
    close_daily = df_daily['close'].values
    ema100_daily = pd.Series(close_daily).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align daily EMA100 to 1h timeframe
    ema100_daily_aligned = align_htf_to_ltf(prices, df_daily, ema100_daily)
    
    # Daily trend: bullish when price > EMA100, bearish when price < EMA100
    daily_uptrend = close_daily > ema100_daily
    daily_downtrend = close_daily < ema100_daily
    
    # Align daily trend to 1h timeframe
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_daily, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_daily, daily_downtrend.astype(float))
    
    # Calculate 1-period VWAP (typical price weighted by volume)
    typical_price = (high + low + close) / 3.0
    vwap = typical_price  # For 1-period, VWAP = typical price
    
    # Price deviation from VWAP
    deviation = close - vwap
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for daily EMA100 to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = deviation[i] < 0 and daily_uptrend_aligned[i] > 0.5 and volume_surge[i]
        short_entry = deviation[i] > 0 and daily_downtrend_aligned[i] > 0.5 and volume_surge[i]
        
        # Exit when price returns to VWAP (mean reversion complete)
        long_exit = deviation[i] >= 0 and position == 1
        short_exit = deviation[i] <= 0 and position == -1
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_AVWAP_Reversion_With_DailyTrend"
timeframe = "1h"
leverage = 1.0