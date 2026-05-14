#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_Trend
Hypothesis: Combines daily Camarilla pivot breakouts (R4/S4) with 12h EMA50 trend filter and volume confirmation.
Trades only in direction of higher timeframe trend to avoid counter-trend whipsaws.
Designed for 15-30 trades/year per symbol with high win rate during trends.
Works in bull/bear by following 12h trend direction - avoids counter-trend losses.
"""

from typing import Tuple
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_Pivot_Breakout_Trend"
timeframe = "4h"
leverage = 1.0

def calculate_camarilla(high: float, low: float, close: float) -> Tuple[float, float, float, float]:
    """Calculate Camarilla pivot levels (R3, R4, S3, S4) from previous period's OHLC."""
    pivot = (high + low + close) / 3
    range_ = high - low
    r3 = pivot + (range_ * 1.1 / 2)
    r4 = pivot + (range_ * 1.1)
    s3 = pivot - (range_ * 1.1 / 2)
    s4 = pivot - (range_ * 1.1)
    return r3, r4, s3, s4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    r3_1d = np.full_like(close_1d, np.nan)
    r4_1d = np.full_like(close_1d, np.nan)
    s3_1d = np.full_like(close_1d, np.nan)
    s4_1d = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        r3, r4, s3, s4 = calculate_camarilla(high_1d[i-1], low_1d[i-1], close_1d[i-1])
        r3_1d[i] = r3
        r4_1d[i] = r4
        s3_1d[i] = s3
        s4_1d[i] = s4
    
    # Calculate 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all to 4h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume filter: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > r4_1d_aligned[i]  # Break above R4
        breakdown_down = close[i] < s4_1d_aligned[i]  # Break below S4
        
        # Entry conditions: only trade in direction of 12h trend
        long_entry = breakout_up and volume_filter and uptrend
        short_entry = breakdown_down and volume_filter and downtrend
        
        # Exit conditions: return to opposite S/R level or trend reversal
        long_exit = (close[i] < s3_1d_aligned[i]) or (not uptrend)  # Break below S3 or trend change
        short_exit = (close[i] > r3_1d_aligned[i]) or (not downtrend)  # Break above R3 or trend change
        
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