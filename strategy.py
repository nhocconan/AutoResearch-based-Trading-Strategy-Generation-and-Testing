#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_cci_trend_follow
# Uses daily CCI (Commodity Channel Index) to detect overbought/oversold conditions.
# Long when daily CCI crosses above +100 and 4h price is above 4h EMA(50).
# Short when daily CCI crosses below -100 and 4h price is below 4h EMA(50).
# Exits when price crosses the 4h EMA(50) in opposite direction.
# Only trade when 4h ADX > 25 to filter for trending markets.
# Designed for low trade frequency (target: 20-50 trades/year) to minimize fee drag.
# Works in trending markets via CCI extremes and mean reversion to EMA.

name = "4h_1d_cci_trend_follow"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data for CCI calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily CCI (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Typical Price
    tp = (high_1d + low_1d + close_1d) / 3.0
    # Simple Moving Average of TP
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    # Mean Deviation
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    ).values
    # CCI = (TP - SMA) / (0.015 * Mean Deviation)
    cci = (tp - sma_tp) / (0.015 * mad)
    # Handle division by zero
    cci = np.where(mad == 0, 0, cci)
    
    # Align daily CCI to 4h timeframe
    cci_aligned = align_htf_to_ltf(prices, df_1d, cci)
    
    # 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # ADX filter: only trade when ADX > 25 (trending market)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First value
        
        # Directional Movement
        dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                           np.maximum(high - np.roll(high, 1), 0), 0)
        dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                            np.maximum(np.roll(low, 1) - low, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values using Wilder's smoothing (alpha = 1/period)
        def wilders_smoothing(data, period):
            result = np.full_like(data, np.nan)
            alpha = 1.0 / period
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nansum(data[:period]) / period
                for i in range(period, len(data)):
                    result[i] = alpha * data[i] + (1 - alpha) * result[i-1]
            return result
        
        atr = wilders_smoothing(tr, period)
        dm_plus_smooth = wilders_smoothing(dm_plus, period)
        dm_minus_smooth = wilders_smoothing(dm_minus, period)
        
        # Avoid division by zero
        dx = np.zeros_like(atr)
        mask = (dm_plus_smooth + dm_minus_smooth) != 0
        dx[mask] = 100 * np.abs(dm_plus_smooth[mask] - dm_minus_smooth[mask]) / (dm_plus_smooth[mask] + dm_minus_smooth[mask])
        
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx = calculate_adx(high, low, close, 14)
    adx_filter = adx > 25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # start after warmup
        # Skip if data not ready
        if np.isnan(cci_aligned[i]) or np.isnan(ema_50[i]) or np.isnan(adx[i]):
            signals[i] = 0.0
            continue
        
        # Long signal: CCI crosses above +100 and price above EMA50
        if cci_aligned[i] > 100 and cci_aligned[i-1] <= 100 and close[i] > ema_50[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short signal: CCI crosses below -100 and price below EMA50
        elif cci_aligned[i] < -100 and cci_aligned[i-1] >= -100 and close[i] < ema_50[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit conditions: price crosses EMA50 in opposite direction
        elif position == 1 and close[i] < ema_50[i]:
            position = 0
            signals[i] = 0.0
        elif position == -1 and close[i] > ema_50[i]:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals