#!/usr/bin/env python3
"""
4h_ADX_Trend_RSI_MeanReversion_v1
Hypothesis: Combines ADX trend strength with RSI mean-reversion on 4h timeframe.
In trending markets (ADX > 25), trades pullbacks to EMA21 in trend direction.
In ranging markets (ADX < 20), trades RSI extremes (30/70) for mean reversion.
Uses volume confirmation to filter false signals. Designed for low trade frequency
(20-30 trades/year) to avoid fee drag while adapting to bull/bear markets.
"""

name = "4h_ADX_Trend_RSI_MeanReversion_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from math import isnan

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- ADX Calculation (14-period) ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period has no previous close
    
    # Directional Movement
    up_move = high - np.roll(high, 1)
    down_move = np.roll(low, 1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed values
    def _wilder_smoothing(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.nansum(arr[:period]) / period
        # Subsequent values: Wilder smoothing
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = _wilder_smoothing(tr, 14)
    plus_di = 100 * _wilder_smoothing(plus_dm, 14) / atr
    minus_di = 100 * _wilder_smoothing(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = _wilder_smoothing(dx, 14)
    
    # --- EMA21 for pullback entries ---
    close_series = pd.Series(close)
    ema_21 = close_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # --- RSI (14-period) ---
    delta = np.diff(close)
    delta = np.insert(delta, 0, 0)  # First element has no delta
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = _wilder_smoothing(gain, 14)
    avg_loss = _wilder_smoothing(loss, 14)
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # --- Volume Confirmation (1.5x 20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (isnan(adx[i]) or isnan(ema_21[i]) or isnan(rsi[i]) or 
            isnan(vol_confirm[i]) or isnan(plus_di[i]) or isnan(minus_di[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine market regime
        is_trending = adx[i] > 25
        is_ranging = adx[i] < 20
        
        if position == 0:
            # Look for new entries
            if is_trending:
                # Trend mode: trade pullbacks to EMA21
                if plus_di[i] > minus_di[i]:  # Uptrend
                    if low[i] <= ema_21[i] and vol_confirm[i]:
                        signals[i] = 0.25
                        position = 1
                else:  # Downtrend
                    if high[i] >= ema_21[i] and vol_confirm[i]:
                        signals[i] = -0.25
                        position = -1
            elif is_ranging:
                # Range mode: trade RSI extremes
                if rsi[i] < 30 and vol_confirm[i]:  # Oversold
                    signals[i] = 0.25
                    position = 1
                elif rsi[i] > 70 and vol_confirm[i]:  # Overbought
                    signals[i] = -0.25
                    position = -1
        else:
            # Manage existing position
            if position == 1:
                # Exit long: RSI overbought or trend reversal
                if rsi[i] > 70 or (plus_di[i] < minus_di[i] and adx[i] > 25):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: RSI oversold or trend reversal
                if rsi[i] < 30 or (plus_di[i] > minus_di[i] and adx[i] > 25):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals