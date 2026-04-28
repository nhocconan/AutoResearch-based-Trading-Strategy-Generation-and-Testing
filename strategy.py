#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
Hypothesis: Uses 1h Candlesticks for entry timing with 4h-based Camarilla R1/S1 (0.5x range) breakouts, filtered by 4h EMA20 trend direction and volume spikes (1.5x 24-bar average). Trades only during 08-20 UTC to avoid low-liquidity periods. Designed for 15-37 trades/year (~60-150 total over 4 years) to minimize fee drag while capturing high-probability breakouts aligned with higher timeframe trend. Works in bull/bear by following 4h trend.
"""

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
    volume = prices['volume'].values
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 4h Camarilla R1/S1 (0.5x range)
    typical_price_4h = (df_4h['high'] + df_4h['low'] + df_4h['close']) / 3
    range_4h = df_4h['high'] - df_4h['low']
    R1 = typical_price_4h + (range_4h * 0.5 / 4)
    S1 = typical_price_4h - (range_4h * 0.5 / 4)
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1.values)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1.values)
    
    # Volume confirmation: >1.5x 24-period MA (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Wait for volume MA to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_20_4h_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_24[i]) or
            not in_session[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 4h EMA20
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Volume confirmation (>1.5x average)
        vol_confirm = volume[i] > (1.5 * vol_ma_24[i])
        
        # Breakout conditions at R1/S1
        long_breakout = close[i] > R1_aligned[i] and vol_confirm and uptrend
        short_breakout = close[i] < S1_aligned[i] and vol_confirm and downtrend
        
        # Exit conditions: return to midpoint of R1/S1
        midpoint = (R1_aligned[i] + S1_aligned[i]) / 2
        long_exit = close[i] < midpoint
        short_exit = close[i] > midpoint
        
        if long_breakout and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_breakout and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0