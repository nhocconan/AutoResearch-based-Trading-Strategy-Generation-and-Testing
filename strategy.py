#!/usr/bin/env python3
"""
1d_1w_Weekly_Trend_With_Daily_Pullback_v1
Hypothesis: Trade weekly trend direction with daily pullback entries. Uses weekly EMA20 for trend and daily RSI(14) for pullback entries.
Designed for low trade frequency (10-25/year) to minimize fee drag while capturing major trends in both bull and bear markets.
Weekly trend filter reduces whipsaws, daily RSI provides precise entry during pullbacks.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Weekly_Trend_With_Daily_Pullback_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter (smooth, fewer whipsaws)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Daily RSI(14) for pullback entries
    delta = pd.Series(close).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.fillna(50).values  # Neutral RSI when insufficient data
    
    # Volume filter: above average to avoid low-liquidity noise
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after RSI warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi_values[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume filter: current volume > 1.2x 20-day average
        volume_filter = volume[i] > 1.2 * vol_ma_20[i]
        
        # Weekly trend direction
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Daily RSI conditions for pullback entries
        rsi_oversold = rsi_values[i] < 30  # Pullback in uptrend
        rsi_overbought = rsi_values[i] > 70  # Pullback in downtrend
        
        # Entry conditions: trade with weekly trend on daily pullback
        long_entry = weekly_uptrend and rsi_oversold and volume_filter
        short_entry = weekly_downtrend and rsi_overbought and volume_filter
        
        # Exit conditions: reverse signal or RSI returns to neutral
        long_exit = (not weekly_uptrend) or (rsi_values[i] > 50)  # Trend change or RSI > 50
        short_exit = (not weekly_downtrend) or (rsi_values[i] < 50)  # Trend change or RSI < 50
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals