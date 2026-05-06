#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy combining 1-month Donchian breakout with 1-day ATR-based volatility filter and volume confirmation
# Donchian(20) on 1-month timeframe identifies major trend direction and breakout levels
# ATR-based volatility filter ensures trades only during sufficient volatility (avoids chop)
# Volume confirmation requires >1.5x 20-period average to validate breakout strength
# Works in bull/bear markets: breakouts capture trends, volatility filter avoids false signals in low volatility
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_1MonthDonchian_ATRVol_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-month (approx 20-day) Donchian channels ONCE before loop
    df_1M = get_htf_data(prices, '1d')  # Using daily data to approximate 1-month (20 days)
    
    if len(df_1M) < 20:
        return np.zeros(n)
    
    # 20-period Donchian high and low (approx 1 month)
    high_20 = pd.Series(df_1M['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1M['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1M, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1M, low_20)
    
    # ATR-based volatility filter: ATR(10) > 0.5 * ATR(30) ensures sufficient volatility
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    
    atr_10 = pd.Series(true_range).rolling(window=10, min_periods=10).mean().values
    atr_30 = pd.Series(true_range).rolling(window=30, min_periods=30).mean().values
    vol_filter = atr_10 > (0.5 * atr_30)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_filter[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 20-day Donchian high with vol and volume confirmation
            if close[i] > donchian_high[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below 20-day Donchian low with vol and volume confirmation
            elif close[i] < donchian_low[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below 20-day Donchian low (failed breakout) or reverse signal
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above 20-day Donchian high (failed breakdown) or reverse signal
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals