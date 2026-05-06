#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1-day Donchian breakouts with volume confirmation and ATR-based risk management
# Breakouts above prior day's high or below prior day's low with volume > 2x 20-period average
# Trend filter: 50-period EMA on 4h timeframe to align with higher timeframe direction
# Works in bull/bear markets: breakouts capture momentum, EMA filter avoids counter-trend trades
# Target: 50-150 total trades over 4 years (12-38/year) with 0.25 position sizing

name = "4h_DailyDonchian_Breakout_VolumeTrend_v1"
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
    
    # Previous day's high and low for Donchian breakout
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Align daily levels to 4h timeframe
    daily_high = align_htf_to_ltf(prices, df_1d, prev_high)
    daily_low = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation: >2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma_20)
    
    # Trend filter: 50-period EMA on 4h timeframe
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    uptrend = close > ema_50
    downtrend = close < ema_50
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(daily_high[i]) or np.isnan(daily_low[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(ema_50[i]) or
            not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above daily high with volume confirmation and uptrend
            if close[i] > daily_high[i] and volume_filter[i] and uptrend[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below daily low with volume confirmation and downtrend
            elif close[i] < daily_low[i] and volume_filter[i] and downtrend[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below daily low (stop loss) or reaches 1.5x daily range (take profit)
            daily_range = daily_high[i] - daily_low[i]
            if close[i] < daily_low[i] or close[i] > daily_low[i] + 1.5 * daily_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above daily high (stop loss) or goes below 1.5x daily range from high (take profit)
            daily_range = daily_high[i] - daily_low[i]
            if close[i] > daily_high[i] or close[i] < daily_high[i] - 1.5 * daily_range:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals