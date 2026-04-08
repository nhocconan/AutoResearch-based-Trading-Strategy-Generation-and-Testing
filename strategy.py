#!/usr/bin/env python3
"""
1h_4h_1d_rsi_trend_v1
Hypothesis: RSI momentum with 4h/1d trend filter to avoid counter-trend trades.
- Entry: RSI(14) crosses above 50 (long) or below 50 (short) on 1h
- Trend filter: Price above/below 4h EMA(50) and 1d EMA(50) for trend alignment
- Session filter: Only trade 08:00-20:00 UTC to reduce noise
- Position sizing: 0.20 for long/short
- Target: 15-35 trades/year (60-140 total over 4 years) to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_trend_v1"
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
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_4h_up = close_4h > ema_50_4h
    trend_4h_down = close_4h < ema_50_4h
    
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
    
    # RSI(14) on 1h
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values  # Fill NaN with neutral 50
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available or outside session
        if (np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(rsi[i]) or not session_filter[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI crosses below 50 OR 4h trend turns down OR 1d trend turns down
            if (rsi[i] < 50) or trend_4h_down_aligned[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI crosses above 50 OR 4h trend turns up OR 1d trend turns up
            if (rsi[i] > 50) or trend_4h_up_aligned[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI crosses above 50 + 4h uptrend + 1d uptrend + session
            if (rsi[i] > 50) and (rsi[i-1] <= 50) and trend_4h_up_aligned[i] and trend_1d_up_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: RSI crosses below 50 + 4h downtrend + 1d downtrend + session
            elif (rsi[i] < 50) and (rsi[i-1] >= 50) and trend_4h_down_aligned[i] and trend_1d_down_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals