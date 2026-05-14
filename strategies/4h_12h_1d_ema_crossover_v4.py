#!/usr/bin/env python3
"""
4h_12h_1d_ema_crossover_v4
Hypothesis: Trend following with EMA crossovers on multiple timeframes.
- Primary: 4h EMA(21) vs EMA(50) for entry/exit
- Trend filter: 12h EMA(50) direction (bullish if close > EMA, bearish if close < EMA)
- Higher timeframe filter: 1d EMA(50) to avoid counter-trend trades in strong trends
- Volume confirmation: 4h volume > 1.5x 20-period average to avoid false breakouts
- Position sizing: 0.25 for long, -0.25 for short
- Target: 20-50 trades/year (80-200 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_1d_ema_crossover_v4"
timeframe = "4h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_12h_up = close_12h > ema_50_12h
    trend_12h_down = close_12h < ema_50_12h
    
    # Forward fill trend
    trend_12h_up_series = pd.Series(trend_12h_up)
    trend_12h_down_series = pd.Series(trend_12h_down)
    trend_12h_up_ffilled = trend_12h_up_series.ffill().values
    trend_12h_down_ffilled = trend_12h_down_series.ffill().values
    
    # Align 12h trend to 4h
    trend_12h_up_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_up_ffilled)
    trend_12h_down_aligned = align_htf_to_ltf(prices, df_12h, trend_12h_down_ffilled)
    
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
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # 4h EMA crossovers
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_12h_up_aligned[i]) or np.isnan(trend_12h_down_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA cross down OR 12h trend turns down OR 1d trend turns down
            if (ema_21[i] < ema_50[i]) or trend_12h_down_aligned[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: EMA cross up OR 12h trend turns up OR 1d trend turns up
            if (ema_21[i] > ema_50[i]) or trend_12h_up_aligned[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: EMA cross up + 12h uptrend + 1d uptrend + volume
            if (ema_21[i] > ema_50[i]) and trend_12h_up_aligned[i] and trend_1d_up_aligned[i] and volume_filter[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: EMA cross down + 12h downtrend + 1d downtrend + volume
            elif (ema_21[i] < ema_50[i]) and trend_12h_down_aligned[i] and trend_1d_down_aligned[i] and volume_filter[i]:
                position = -1
                signals[i] = -0.25
    
    return signals