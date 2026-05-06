#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian channel breakouts with volume confirmation and 1d trend filter
# Breakouts above 4h upper band or below lower band with volume > 2x 20-period average indicate momentum
# Trend filter: 1d EMA50 to ensure trades align with higher timeframe trend
# Volume confirmation reduces false breakouts; trend filter avoids counter-trend trades
# Works in bull/bear markets: breakouts capture trends, filter improves win rate in ranging markets
# Target: 80-150 total trades over 4 years (20-38/year) with 0.20 position sizing

name = "1h_DonchianBreakout_VolumeTrend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channel (20-period high/low) ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 20-period high and low for Donchian channel
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align 4h Donchian levels to 1h timeframe
    upper_band = align_htf_to_ltf(prices, df_4h, high_20)
    lower_band = align_htf_to_ltf(prices, df_4h, low_20)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Trend filter: 1d EMA50
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend = close > ema_50_1d_aligned
    downtrend = close < ema_50_1d_aligned
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50_1d_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above 4h upper band with volume confirmation and uptrend
            if close[i] > upper_band[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.20
                position = 1
            # Short breakout: price breaks below 4h lower band with volume confirmation and downtrend
            elif close[i] < lower_band[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price breaks below 4h lower band (failed breakout) or reaches opposite band (take profit)
            if close[i] < lower_band[i] or close[i] > upper_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price breaks above 4h upper band (failed breakdown) or reaches opposite band (take profit)
            if close[i] > upper_band[i] or close[i] < lower_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals