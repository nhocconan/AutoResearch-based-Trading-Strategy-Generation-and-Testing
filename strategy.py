#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using daily Donchian breakouts with volume confirmation and trend filter
# Daily Donchian(20) provides key structural levels for breakouts
# Breakout above upper band or below lower band with volume > 2.0x 20-period average indicates strong momentum
# Trend filter: 20-period EMA on 12h timeframe to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, volatility filter avoids whipsaws
# Target: 50-150 total trades over 4 years (12-37/year) with 0.25 position sizing

name = "12h_Donchian20_1d_VolumeTrendFilter_v1"
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
    
    # Calculate daily Donchian channels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20) - using previous day's data to avoid look-ahead
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align daily Donchian levels to 12h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Volume confirmation: >2.0x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Trend filter: 20-period EMA on 12h timeframe
    close_series = pd.Series(close)
    ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend = close > ema_20
    downtrend = close < ema_20
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_band_aligned[i]) or np.isnan(lower_band_aligned[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_20[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above upper band with volume confirmation and uptrend
            if close[i] > upper_band_aligned[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below lower band with volume confirmation and downtrend
            elif close[i] < lower_band_aligned[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below lower band (failed support) or reaches upper band (take profit)
            if close[i] < lower_band_aligned[i] or close[i] > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band (failed resistance) or reaches lower band (take profit)
            if close[i] > upper_band_aligned[i] or close[i] < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals