#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 1d Donchian Channel Breakout with 1w Trend Filter
# Hypothesis: Donchian(20) breakouts capture strong trends. Weekly EMA filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts in choppy markets. Works in bull (breakouts continue) and bear (breakdowns continue) by following the weekly trend.
# Target: 15-25 trades/year (60-100 over 4 years) to avoid fee drag.
name = "1d_donchian20_1w_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Donchian Channel (20-period high/low) on 1d
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max()
    donchian_low = low_series.rolling(window=20, min_periods=20).min()
    
    # 1-week EMA(50) for trend filter
    weekly_close = df_1w['close'].values
    weekly_ema = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    weekly_ema_1d = align_htf_to_ltf(prices, df_1w, weekly_ema)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_ema_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below Donchian low (trend reversal)
            if close[i] < donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: price breaks above Donchian high (trend reversal)
            if close[i] > donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long: price breaks above Donchian high with weekly uptrend
            if close[i] > donchian_high[i] and close[i] > weekly_ema_1d[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below Donchian low with weekly downtrend
            elif close[i] < donchian_low[i] and close[i] < weekly_ema_1d[i]:
                position = -1
                signals[i] = -0.25
    
    return signals