#!/usr/bin/env python3
# 4h_price_action_channel_v1
# Hypothesis: Price channel breakout with volume confirmation and trend filter.
# Long: Price breaks above 4h Donchian(20) high + volume > 1.5x 20-period avg + 1d close > 1d SMA50
# Short: Price breaks below 4h Donchian(20) low + volume > 1.5x 20-period avg + 1d close < 1d SMA50
# Exit: Price crosses back through Donchian midpoint or volume drops below average.
# Target: 20-40 trades/year per symbol for low fee drag and robust performance in bull/bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_action_channel_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channel (20-period)
    donch_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_period, min_periods=donch_period).max().values
    donch_low = low_series.rolling(window=donch_period, min_periods=donch_period).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    close_series = pd.Series(close)
    vol_ma = close_series.rolling(window=vol_ma_period, min_periods=vol_ma_period).mean().values
    vol_surge = volume > (1.5 * vol_ma)
    vol_surge[:vol_ma_period-1] = False  # Not enough data for MA
    
    # Get 1d data for trend filter (close vs SMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma50_1d = pd.Series(close_1d).rolling(window=50, min_periods=50).mean().values
    # Trend: 1 if close > SMA50, -1 if close < SMA50
    trend_1d = np.where(close_1d > sma50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_period, vol_ma_period, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(trend_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below midpoint or volume drops below average
            if close[i] < donch_mid[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above midpoint or volume drops below average
            if close[i] > donch_mid[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high, volume surge, 1d uptrend
            if (close[i] > donch_high[i] and 
                vol_surge[i] and 
                trend_1d_aligned[i] > 0):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, volume surge, 1d downtrend
            elif (close[i] < donch_low[i] and 
                  vol_surge[i] and 
                  trend_1d_aligned[i] < 0):
                position = -1
                signals[i] = -0.25
    
    return signals