#!/usr/bin/env python3
"""
1d Bollinger Band Mean Reversion + Weekly Trend Filter
Hypothesis: In ranging markets, price reverts to mean at Bollinger Bands.
In trending markets (weekly trend), only trade pullbacks in trend direction.
Exit at mean (BB middle) or stoploss. Works in bull (buy dips) and bear (sell rallies).
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_bb_mean_reversion_weekly_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load weekly data for trend filter (once before loop)
    df_weekly = get_htf_data(prices, '1w')
    close_weekly = df_weekly['close'].values
    
    # Calculate 50-period EMA for weekly trend filter
    ema_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Bollinger Bands (20, 2)
    bb_period = 20
    bb_std = 2
    sma = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).mean().values
    std = pd.Series(close).rolling(window=bb_period, min_periods=bb_period).std().values
    upper = sma + (std * bb_std)
    lower = sma - (std * bb_std)
    
    # Bollinger Band Width for regime filter (low volatility = range)
    bb_width = (upper - lower) / (sma + 1e-10)
    # Calculate percentile of BB width over 50 periods to identify ranging markets
    bb_width_series = pd.Series(bb_width)
    bb_width_percentile = bb_width_series.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5, raw=False
    ).values
    # Range market: BB width in lower 40th percentile (low volatility)
    range_filter = bb_width_percentile <= 0.4
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(bb_period, 50) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_weekly_aligned[i]) or np.isnan(sma[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches middle (mean reversion) OR stoploss
            if close[i] >= sma[i] or close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches middle (mean reversion) OR stoploss
            if close[i] <= sma[i] or close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price at BB extreme + in range market (low volatility)
            at_upper = close[i] >= upper[i]
            at_lower = close[i] <= lower[i]
            
            if range_filter[i]:
                # In ranging market: mean reversion at extremes
                if at_lower:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif at_upper:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                # In trending market: only trade pullbacks in trend direction
                # Weekly trend up: look for pullbacks to lower BB
                # Weekly trend down: look for rallies to upper BB
                weekly_uptrend = close[i] > ema_weekly_aligned[i]
                if weekly_uptrend and at_lower:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                elif (not weekly_uptrend) and at_upper:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
    
    return signals
</ly>