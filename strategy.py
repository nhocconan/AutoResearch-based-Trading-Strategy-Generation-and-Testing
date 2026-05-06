#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with daily ATR filter and volume confirmation
# Uses Donchian(20) channels from 12h timeframe for breakouts
# Filters: daily ATR(14) < 0.05 * price (low volatility regime) and volume > 1.5x 20-period average
# Works in bull/bear markets: breakouts capture trends, volatility filter avoids choppy markets
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_DailyATR_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range calculation for daily
    prev_close_1d = df_1d['close'].shift(1)
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - prev_close_1d)
    tr3 = abs(df_1d['low'] - prev_close_1d)
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_14 = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Align daily ATR to 12h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    # Volatility filter: ATR < 5% of price (low volatility regime)
    vol_filter = atr_14_aligned < (0.05 * close)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_14_aligned[i]) or np.isnan(volume_filter[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above Donchian high with vol filter and volume confirmation
            if close[i] > donchian_high[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below Donchian low with vol filter and volume confirmation
            elif close[i] < donchian_low[i] and vol_filter[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals