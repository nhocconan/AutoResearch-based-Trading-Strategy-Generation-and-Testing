#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using weekly Bollinger Bands with volume confirmation and trend filter
# Weekly Bollinger Bands (20, 2) provide dynamic support/resistance levels
# Breakout above upper band or below lower band with volume > 1.8x 20-period average indicates strong momentum
# Trend filter: 20-period EMA on 4h timeframe to avoid counter-trend trades
# Works in bull/bear markets: breakouts capture trends, reversals capture pullbacks within trend
# Target: 75-200 total trades over 4 years (19-50/year) with 0.25 position sizing

name = "4h_WeeklyBB20_2_VolumeTrendFilter_v1"
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
    
    # Calculate weekly Bollinger Bands ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly close for Bollinger Bands
    weekly_close = df_1w['close'].values
    close_series = pd.Series(weekly_close)
    
    # Bollinger Bands (20, 2)
    sma_20 = close_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_series.rolling(window=20, min_periods=20).std().values
    upper_band = sma_20 + (2 * std_20)
    lower_band = sma_20 - (2 * std_20)
    
    # Align weekly bands to 4h timeframe
    upper_band_aligned = align_htf_to_ltf(prices, df_1w, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1w, lower_band)
    
    # Volume confirmation: >1.8x 20-period average (higher threshold to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.8 * vol_ma_20)
    
    # Trend filter: 20-period EMA on 4h timeframe
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
            # Exit long: price breaks below lower band (failed breakout) or reaches upper band (take profit)
            if close[i] < lower_band_aligned[i] or close[i] > upper_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above upper band (failed breakdown) or reaches lower band (take profit)
            if close[i] > upper_band_aligned[i] or close[i] < lower_band_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals