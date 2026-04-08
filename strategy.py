#!/usr/bin/env python3
# 1d_1w_donchian20_volume_sma_filter_v1
# Hypothesis: Donchian channel breakout on 1d with weekly trend filter and volume confirmation.
# Long when price breaks above 20-day high with price above weekly SMA50 and volume > 1.5x average.
# Short when price breaks below 20-day low with price below weekly SMA50 and volume > 1.5x average.
# Exit when price returns to 20-day moving average or volume drops below average.
# Target: 7-25 trades/year (30-100 total over 4 years) to avoid overtrading.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_donchian20_volume_sma_filter_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily Donchian channel (20)
    donch_len = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_len, min_periods=donch_len).max().values
    donch_low = low_series.rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Daily 20-period SMA for exit
    close_series = pd.Series(close)
    sma20 = close_series.rolling(window=20, min_periods=20).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    # Weekly trend filter: SMA50 on 1w timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        sma50_1w = np.full(len(close), np.nan)
    else:
        close_1w = pd.Series(df_1w['close'].values)
        sma50_1w_raw = close_1w.rolling(window=50, min_periods=50).mean().values
        sma50_1w = align_htf_to_ltf(prices, df_1w, sma50_1w_raw)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(donch_len, 20, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(sma20[i]) or np.isnan(vol_ma[i]) or np.isnan(sma50_1w[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price below 20-day SMA or volume drops below average
            if close[i] < sma20[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Price above 20-day SMA or volume drops below average
            if close[i] > sma20[i] or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: Price above 20-day high with price above weekly SMA50 and volume surge
            if (close[i] > donch_high[i] and close[i] > sma50_1w[i] and vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price below 20-day low with price below weekly SMA50 and volume surge
            elif (close[i] < donch_low[i] and close[i] < sma50_1w[i] and vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals