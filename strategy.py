#!/usr/bin/env python3
"""
12h_VWAP_RSI_Momentum_v1
Hypothesis: Momentum strategy combining VWAP mean reversion with RSI momentum on 12h timeframe.
- Entry: Price crosses above VWAP with RSI > 55 for long, crosses below VWAP with RSI < 45 for short
- Filter: Only trade when 1d trend aligns (1d EMA50 slope) to avoid counter-trend trades
- Volume: Require volume > 1.3x 20-period average for confirmation
- Exit: Opposite VWAP cross or RSI crosses 50 (middle)
- Position sizing: 0.25
- Target: 15-35 trades/year (60-140 total over 4 years) - fits 12h sweet spot
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_VWAP_RSI_Momentum_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # VWAP calculation (typical price * volume cumulative)
    typical_price = (high + low + close) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume)
    vwap_denominator = np.cumsum(volume)
    vwap = vwap_numerator / vwap_denominator
    
    # RSI calculation (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume filter
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA(50) slope for trend direction
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_slope = ema_50_1d - np.roll(ema_50_1d, 1)
    ema_slope[0] = 0
    trend_1d_up = ema_slope > 0
    trend_1d_down = ema_slope < 0
    
    # Forward fill trend
    trend_1d_up_series = pd.Series(trend_1d_up)
    trend_1d_down_series = pd.Series(trend_1d_down)
    trend_1d_up_ffilled = trend_1d_up_series.ffill().values
    trend_1d_down_ffilled = trend_1d_down_series.ffill().values
    
    # Align 1d trend to 12h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up_ffilled)
    trend_1d_down_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_down_ffilled)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(vwap[i]) or np.isnan(rsi[i]) or np.isnan(volume_filter[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(trend_1d_down_aligned[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Price crosses below VWAP OR RSI crosses below 50
            if close[i] < vwap[i] or rsi[i] < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: Price crosses above VWAP OR RSI crosses above 50
            if close[i] > vwap[i] or rsi[i] > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: Price crosses above VWAP + RSI > 55 + 1d uptrend + volume
            if (close[i] > vwap[i] and close[i-1] <= vwap[i-1] and  # Cross above VWAP
                rsi[i] > 55 and trend_1d_up_aligned[i] and volume_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price crosses below VWAP + RSI < 45 + 1d downtrend + volume
            elif (close[i] < vwap[i] and close[i-1] >= vwap[i-1] and  # Cross below VWAP
                  rsi[i] < 45 and trend_1d_down_aligned[i] and volume_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals