#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h trend filter (EMA50) and session filter (08-20 UTC).
- Primary timeframe: 1h for precise entry/exit timing.
- HTF: 4h EMA50 for trend direction (bullish if close > EMA50, bearish if close < EMA50).
- Session: Only trade between 08:00-20:00 UTC to avoid low-liquidity hours.
- Entry: Long when price breaks above H3 level AND 4h trend bullish AND in session.
         Short when price breaks below L3 level AND 4h trend bearish AND in session.
- Exit: Opposite breakout (price breaks below H3 for long, above L3 for short) or session end.
- Signal size: 0.20 discrete to limit drawdown and reduce fee churn.
- Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe.
Camarilla pivot levels provide precise intraday support/resistance. Works in ranging markets via mean reversion at H3/L3 and in trends via breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate Camarilla pivot levels (based on previous day)
    # Typical price for pivot calculation
    typical_price = (high + low + close) / 3
    # Use previous day's typical price for today's pivot levels
    prev_typical = np.roll(typical_price, 1)
    prev_typical[0] = np.nan  # First value has no previous day
    
    # Calculate pivot and support/resistance levels
    pivot = prev_typical
    # Camarilla levels: H3/L3 are the key breakout levels
    h3 = pivot + (1.1 * (high - low) / 2)
    l3 = pivot - (1.1 * (high - low) / 2)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema_50 = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    # Trend: 1 if bullish (close > EMA50), -1 if bearish (close < EMA50), 0 otherwise
    trend_4h = np.where(close_4h > ema_50, 1, np.where(close_4h < ema_50, -1, 0))
    
    # Align HTF indicators to 1h
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Session filter: 08:00-20:00 UTC
    # open_time is already datetime64[ns], access via index
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 50  # Need enough bars for 4h EMA50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h3[i]) or np.isnan(l3[i]) or 
            np.isnan(trend_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        in_sess = in_session[i]
        trend = trend_4h_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if in_sess:
                # Bullish: price breaks above H3 AND 4h trend bullish
                if curr_high > h3[i] and trend == 1:
                    signals[i] = 0.20
                    position = 1
                # Bearish: price breaks below L3 AND 4h trend bearish
                elif curr_low < l3[i] and trend == -1:
                    signals[i] = -0.20
                    position = -1
        elif position == 1:
            # Long exit: price breaks below H3 OR session ends
            if curr_low < h3[i] or not in_sess:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above L3 OR session ends
            if curr_high > l3[i] or not in_sess:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_Session_v1"
timeframe = "1h"
leverage = 1.0