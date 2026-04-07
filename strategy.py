#!/usr/bin/env python3
"""
4h_adx_rsi_range_mean_reversion_v1
Hypothesis: On 4-hour timeframe, use ADX to identify ranging markets (ADX < 25) and RSI for mean reversion entries.
Long when RSI < 30 and price > EMA(50) in ranging markets.
Short when RSI > 70 and price < EMA(50) in ranging markets.
Exit when RSI returns to neutral range (40-60).
Designed for 20-40 trades/year to minimize fee drag while capturing mean reversion in ranging markets.
Works in both bull/bear markets as ADX filter avoids trending periods where mean reversion fails.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_adx_rsi_range_mean_reversion_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for trend strength
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    up_move = np.concatenate([[0], high[1:] - high[:-1]])
    down_move = np.concatenate([[0], low[:-1] - low[1:]])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smooth_wilder(arr, period):
        smoothed = np.zeros_like(arr)
        smoothed[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            smoothed[i] = (smoothed[i-1] * (period-1) + arr[i]) / period
        return smoothed
    
    tr_smooth = smooth_wilder(tr, 14)
    plus_dm_smooth = smooth_wilder(plus_dm, 14)
    minus_dm_smooth = smooth_wilder(minus_dm, 14)
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = np.zeros_like(dx)
    adx[13:] = smooth_wilder(dx[13:], 14) if np.sum(~np.isnan(dx[13:])) >= 14 else np.nan
    
    # RSI(14)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    avg_gain[13] = np.mean(gain[:14])
    avg_loss[13] = np.mean(loss[:14])
    for i in range(14, len(close)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # EMA(50) for trend filter within range
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(adx[i]) or np.isnan(rsi[i]) or 
            np.isnan(ema_50[i])):
            signals[i] = 0.0
            continue
            
        # Range condition: ADX < 25 indicates ranging market
        is_ranging = adx[i] < 25
        
        if position == 1:  # Long position
            # Exit: RSI returns to neutral range (40-60)
            if rsi[i] >= 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI returns to neutral range (40-60)
            if rsi[i] <= 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter in ranging markets
            if is_ranging:
                # Long: RSI oversold (<30) and price above EMA(50)
                if rsi[i] < 30 and close[i] > ema_50[i]:
                    position = 1
                    signals[i] = 0.25
                # Short: RSI overbought (>70) and price below EMA(50)
                elif rsi[i] > 70 and close[i] < ema_50[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals