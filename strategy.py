#!/usr/bin/env python3
"""
1h EMA Pullback with 4h Trend + Volume + Session Filter
Hypothesis: Buy pullbacks to EMA20 in bullish 4h trends, sell rallies to EMA20 in bearish 4h trends.
Uses volume confirmation and restricts trading to active session (08-20 UTC) to avoid noise.
Designed for low trade frequency (target: 60-150 total trades over 4 years) to minimize fee drag.
Works in bull markets via trend-following pulls and in bear via counter-trend fades at extremes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_ema_pullback_4htrend_vol_sess_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Session filter: 08:00-20:00 UTC
    # Pre-compute hour from DatetimeIndex (already datetime64[ns])
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 20-period EMA for pullback entries
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False, min_periods=20).values
    
    # 4h EMA50 for trend bias
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).values
    trend_bias_4h = np.where(close_4h > ema_4h, 1, -1)
    trend_bias_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_bias_4h)
    
    # ATR for stop loss
    tr = np.maximum(
        high[1:] - low[1:],
        np.abs(high[1:] - close[:-1]),
        np.abs(low[1:] - close[:-1])
    )
    atr = np.full(n, np.nan)
    if len(tr) > 0:
        atr[1] = tr[0]
        for i in range(2, n):
            atr[i] = (tr[i-1] * 19 + atr[i-1]) / 20  # Wilder smoothing
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_exit = 0
    
    # Start from warmup period
    start = 50  # Need EMA20 and ATR warmup
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema20[i]) or np.isnan(trend_bias_4h_aligned[i]) or 
            np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            bars_since_exit += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price crosses below EMA20 OR stoploss hit
            if (close[i] < ema20[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = 0.20
            bars_since_exit += 1
        elif position == -1:  # short position
            # Exit: price crosses above EMA20 OR stoploss hit
            if (close[i] > ema20[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_exit = 0
            else:
                signals[i] = -0.20
            bars_since_exit += 1
        else:
            # Look for entries - only during session
            if not in_session[i]:
                signals[i] = 0.0
                bars_since_exit += 1
                continue
            
            # Minimum bars since exit to prevent overtrading
            if bars_since_exit < 24:  # At least 1 day between trades
                signals[i] = 0.0
                bars_since_exit += 1
                continue
            
            # Long: pullback to EMA20 in bullish 4h trend with volume
            if (close[i] >= ema20[i] * 0.995 and  # Allow slight dip below EMA
                close[i] <= ema20[i] * 1.005 and   # Allow slight rise above EMA
                trend_bias_4h_aligned[i] == 1 and
                volume_filter):
                signals[i] = 0.20
                position = 1
                entry_price = close[i]
                bars_since_exit = 0
            # Short: rally to EMA20 in bearish 4h trend with volume
            elif (close[i] >= ema20[i] * 0.995 and
                  close[i] <= ema20[i] * 1.005 and
                  trend_bias_4h_aligned[i] == -1 and
                  volume_filter):
                signals[i] = -0.20
                position = -1
                entry_price = close[i]
                bars_since_exit = 0
            else:
                signals[i] = 0.0
                bars_since_exit += 1
    
    return signals