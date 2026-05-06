#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Donchian breakout with volume confirmation and trend filter
# Daily Donchian channels (20-period) provide key support/resistance levels
# Breakout above upper band or below lower band with volume > 1.5x 20-period average indicates strong momentum
# Trend filter: 4h EMA(50) > EMA(100) for longs, EMA(50) < EMA(100) for shorts to align with intermediate trend
# Works in bull/bear markets: breakouts capture trends, with trend filter reducing counter-trend trades
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "4h_DailyDonchian20_VolumeTrendFilter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily Donchian channels (20-period) ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily high and low for Donchian calculation
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on daily data
    high_series = pd.Series(daily_high)
    low_series = pd.Series(daily_low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume confirmation: >1.5x 20-period average (moderate threshold to balance signal quality and frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    # Trend filter: 4h EMA(50) > EMA(100) for uptrend, EMA(50) < EMA(100) for downtrend
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_100 = close_series.ewm(span=100, adjust=False, min_periods=100).mean().values
    uptrend = ema_50 > ema_100
    downtrend = ema_50 < ema_100
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(uptrend[i]) or np.isnan(downtrend[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily Donchian high with volume confirmation and uptrend
            if close[i] > donchian_high_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below daily Donchian low with volume confirmation and downtrend
            elif close[i] < donchian_low_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily Donchian low (failed breakout) or reaches opposite band (take profit)
            if close[i] < donchian_low_aligned[i] or close[i] > donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily Donchian high (failed breakdown) or reaches opposite band (take profit)
            if close[i] > donchian_high_aligned[i] or close[i] < donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals