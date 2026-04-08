#!/usr/bin/env python3
# 1h_4h_1d_volume_breakout_v1
# Hypothesis: 1h breakouts aligned with 4h/1d trend and volume confirmation.
# - Trend filter: 4h EMA(20) direction (bullish if close > EMA, bearish if close < EMA)
# - Higher timeframe filter: 1d EMA(50) to avoid counter-trend trades in strong trends
# - Entry: 1h price breaks above/below 20-period high/low with volume > 1.5x 20-period average
# - Exit: Opposite breakout or trend reversal
# - Position sizing: 0.20 for long, -0.20 for short
# - Session filter: 08-20 UTC to reduce noise trades
# Target: 15-37 trades/year (60-150 total over 4 years)

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_volume_breakout_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # 4h EMA(20) for trend
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_4h_up = close_4h > ema_20_4h
    trend_4h_down = close_4h < ema_20_4h
    
    # Forward fill trend
    trend_4h_up_series = pd.Series(trend_4h_up)
    trend_4h_down_series = pd.Series(trend_4h_down)
    trend_4h_up_ffilled = trend_4h_up_series.ffill().values
    trend_4h_down_ffilled = trend_4h_down_series.ffill().values
    
    # Align 4h trend to 1h
    trend_4h_up_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_up_ffilled)
    trend_4h_down_aligned = align_htf_to_ltf(prices, df_4h, trend_4h_down_ffilled)
    
    # Get 1d data for higher timeframe filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) for trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = close_1d > ema_50_1d
    trend_1d_down = close_1d < ema_50_1d
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 1h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # 1h 20-period high/low for breakout
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_filter[i]) or np.isnan(session_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price breaks below 20-period low OR 4h trend turns down OR 1d trend turns down
            if (low[i] < low_20[i]) or trend_4h_down_aligned[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Position size
                
        elif position == -1:  # Short position
            # Exit: price breaks above 20-period high OR 4h trend turns up OR 1d trend turns up
            if (high[i] > high_20[i]) or trend_4h_up_aligned[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Position size
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high + 4h uptrend + 1d uptrend + volume + session
            if (high[i] > high_20[i]) and trend_4h_up_aligned[i] and trend_1d_up_aligned[i] and volume_filter[i] and session_filter[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: price breaks below 20-period low + 4h downtrend + 1d downtrend + volume + session
            elif (low[i] < low_20[i]) and trend_4h_down_aligned[i] and trend_1d_down_aligned[i] and volume_filter[i] and session_filter[i]:
                position = -1
                signals[i] = -0.20
    
    return signals