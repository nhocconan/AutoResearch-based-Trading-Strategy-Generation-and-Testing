#!/usr/bin/env python3
"""
6h_1d_rsi_momentum_v1
Hypothesis: Momentum strategy using RSI divergence and trend filters.
- Primary: 6h RSI(14) with overbought/oversold levels (70/30) for mean reversion
- Trend filter: 1d EMA(50) direction (bullish if close > EMA, bearish if close < EMA)
- Momentum confirmation: 6h price > 6h EMA(20) for longs, < EMA(20) for shorts
- Position sizing: 0.25 for long, -0.25 for short
Target: 50-150 total trades over 4 years (12-37/year)
Works in bull markets via trend-following longs, in bear via mean reversion shorts at resistance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_rsi_momentum_v1"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
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
    
    # Align 1d trend to 6h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    # 6h RSI(14) for momentum/mean reversion
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # 6h EMA(20) for momentum confirmation
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            np.isnan(rsi_values[i]) or np.isnan(ema_20[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI overbought OR trend turns down OR price breaks EMA20 down
            if (rsi_values[i] > 70) or trend_1d_down_aligned[i] or (close[i] < ema_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI oversold OR trend turns up OR price breaks EMA20 up
            if (rsi_values[i] < 30) or trend_1d_up_aligned[i] or (close[i] > ema_20[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI oversold + uptrend + price above EMA20
            if (rsi_values[i] < 30) and trend_1d_up_aligned[i] and (close[i] > ema_20[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI overbought + downtrend + price below EMA20
            elif (rsi_values[i] > 70) and trend_1d_down_aligned[i] and (close[i] < ema_20[i]):
                position = -1
                signals[i] = -0.25
    
    return signals