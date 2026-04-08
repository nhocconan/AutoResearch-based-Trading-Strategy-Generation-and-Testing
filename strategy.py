#!/usr/bin/env python3
# 4h_price_channel_momentum_v1
# Hypothesis: Combines 4h Donchian channel breakout with 1d trend filter (price relative to SMA50) and volume confirmation.
# Long when price breaks above Donchian(20) high, price > 1d SMA50, and volume > 1.5x average.
# Short when price breaks below Donchian(20) low, price < 1d SMA50, and volume > 1.5x average.
# Exit when price crosses back through Donchian midline (mean of upper/lower) or volume drops below average.
# Uses tight entry conditions to limit trades and reduce fee drag. Target: 20-40 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_price_channel_momentum_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Get 1d data for trend filter (SMA50)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    sma_period = 50
    sma50_1d = pd.Series(close_1d).rolling(window=sma_period, min_periods=sma_period).mean().values
    sma50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_period, vol_ma_period, sma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(sma50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below Donchian midline or volume drops below average
            if close[i] < donch_mid[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above Donchian midline or volume drops below average
            if close[i] > donch_mid[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price breaks above Donchian high, price > 1d SMA50, volume surge
            if (close[i] > donch_high[i] and 
                close[i] > sma50_1d_aligned[i] and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price breaks below Donchian low, price < 1d SMA50, volume surge
            elif (close[i] < donch_low[i] and 
                  close[i] < sma50_1d_aligned[i] and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals