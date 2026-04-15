# 1h_4h_DailyVolBreakout_TrendFollow
# Hypothesis: Use 1d ATR volatility filter (low volatility = range) and 4h EMA trend filter.
# Enter long when price breaks above 1h high of previous 4 bars in low volatility + uptrend.
# Enter short when price breaks below 1h low of previous 4 bars in low volatility + downtrend.
# Low volatility reduces false breakouts; trend filter ensures directional bias.
# Position size 0.20 to manage drawdown. Session filter 08-20 UTC to avoid low-liquidity hours.
# Expects ~20-40 trades/year per symbol, suitable for 1h timeframe.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for volatility filter
    daily = get_htf_data(prices, '1d')
    daily_high = daily['high'].values
    daily_low = daily['low'].values
    daily_close = daily['close'].values
    
    # Calculate daily ATR for volatility filter
    daily_close_prev = np.concatenate([[daily_close[0]], daily_close[:-1]])
    tr = np.maximum(daily_high - daily_low,
                    np.maximum(np.abs(daily_high - daily_close_prev),
                               np.abs(daily_low - daily_close_prev)))
    atr_daily = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ratio_daily = atr_daily / daily_close  # ATR as fraction of price
    
    # Align daily ATR ratio to 1h timeframe
    atr_ratio_1h = align_htf_to_ltf(prices, daily, atr_ratio_daily)
    
    # Get 4h data for trend filter
    h4 = get_htf_data(prices, '4h')
    h4_close = h4['close'].values
    
    # Calculate 4h EMA(20) for trend
    ema_4h = pd.Series(h4_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, h4, ema_4h)
    
    # Pre-calculate 1h rolling max/min for breakout levels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    roll_max_4 = high_series.rolling(window=4, min_periods=4).max().shift(1).values  # exclude current bar
    roll_min_4 = low_series.rolling(window=4, min_periods=4).min().shift(1).values
    
    # Pre-calculate hour filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(atr_ratio_1h[i]) or np.isnan(ema_4h_aligned[i]) or 
            np.isnan(roll_max_4[i]) or np.isnan(roll_min_4[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volatility filter: only trade in low volatility environments
        if atr_ratio_1h[i] >= 0.015:  # High volatility = avoid breakouts
            signals[i] = 0.0
            continue
        
        # Trend filter: use 4h EMA
        uptrend = close[i] > ema_4h_aligned[i]
        downtrend = close[i] < ema_4h_aligned[i]
        
        # Breakout logic: price breaks recent 4-bar high/low
        if uptrend and close[i] > roll_max_4[i]:
            signals[i] = 0.20  # Long breakout
        elif downtrend and close[i] < roll_min_4[i]:
            signals[i] = -0.20  # Short breakdown
        else:
            signals[i] = 0.0
    
    return signals

name = "1h_4h_DailyVolBreakout_TrendFollow"
timeframe = "1h"
leverage = 1.0