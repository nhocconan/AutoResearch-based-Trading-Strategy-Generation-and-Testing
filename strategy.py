#!/usr/bin/env python3
"""
1h_4h_1d_rsi_mean_reversion_v1
Hypothesis: Mean reversion strategy for 1h timeframe using RSI extremes with multi-timeframe trend filters.
- RSI(14) < 30 for long entry, > 70 for short entry on 1h
- Trend filter: 4h close > EMA(50) for long, < EMA(50) for short
- Higher timeframe filter: 1d close > EMA(50) for long, < EMA(50) for short
- Session filter: 08-20 UTC to avoid low liquidity periods
- Position sizing: 0.20 for both long and short
- Target: 15-30 trades/year (60-120 total over 4 years) to avoid fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_rsi_mean_reversion_v1"
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
    
    # 1h RSI calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
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
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available or outside session
        if (np.isnan(rsi[i]) or np.isnan(trend_4h_up_aligned[i]) or np.isnan(trend_4h_down_aligned[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i]) or
            not session_filter[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI > 50 (mean reversion complete) OR trend changes
            if rsi[i] > 50 or trend_4h_down_aligned[i] or trend_1d_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20  # Position size
                
        elif position == -1:  # Short position
            # Exit: RSI < 50 (mean reversion complete) OR trend changes
            if rsi[i] < 50 or trend_4h_up_aligned[i] or trend_1d_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20  # Position size
        else:  # Flat, look for entry
            # Long entry: RSI < 30 (oversold) + 4h uptrend + 1d uptrend + session
            if (rsi[i] < 30) and trend_4h_up_aligned[i] and trend_1d_up_aligned[i]:
                position = 1
                signals[i] = 0.20
            # Short entry: RSI > 70 (overbought) + 4h downtrend + 1d downtrend + session
            elif (rsi[i] > 70) and trend_4h_down_aligned[i] and trend_1d_down_aligned[i]:
                position = -1
                signals[i] = -0.20
    
    return signals