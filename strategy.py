#!/usr/bin/env python3
"""
6h Linear Regression Channel + Volume Confirmation
Hypothesis: Linear regression channels provide dynamic support/resistance.
Price tends to revert to mean within channel; breakouts indicate trend.
Long when price near lower band with bullish momentum, short when near upper band with bearish momentum.
Uses 1d trend filter to avoid counter-trend trades. Works in bull (buy dips in uptrend) and bear (sell rallies in downtrend).
Target: 60-120 total trades over 4 years (15-30/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "14431_6h_linear_regression_channel_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Linear regression channel (60 periods)
    lr_period = 60
    
    def linreg_intercept(arr, period):
        """Calculate linear regression intercept"""
        if len(arr) < period:
            return np.nan
        x = np.arange(period)
        y = arr[-period:]
        if np.std(y) == 0:
            return y[-1]
        slope = np.polyfit(x, y, 1)[0]
        intercept = y[-1] - slope * (period - 1)
        return intercept
    
    def linreg_slope(arr, period):
        """Calculate linear regression slope"""
        if len(arr) < period:
            return np.nan
        x = np.arange(period)
        y = arr[-period:]
        if np.std(y) == 0:
            return 0
        return np.polyfit(x, y, 1)[0]
    
    # Precompute arrays
    lr_intercept = np.full(n, np.nan)
    lr_slope = np.full(n, np.nan)
    
    for i in range(lr_period - 1, n):
        lr_intercept[i] = linreg_intercept(close[:i+1], lr_period)
        lr_slope[i] = linreg_slope(close[:i+1], lr_period)
    
    # Calculate channel
    x = np.arange(lr_period)
    lr_mid = lr_intercept + lr_slope * (lr_period - 1)  # Current price prediction
    # Standard error of estimate
    lr_std = np.full(n, np.nan)
    for i in range(lr_period - 1, n):
        y_pred = lr_intercept[i] + lr_slope[i] * x
        y_actual = close[i - lr_period + 1:i + 1]
        if len(y_actual) == lr_period:
            lr_std[i] = np.sqrt(np.mean((y_actual - y_pred) ** 2))
    
    # Upper and lower bands (2 standard errors)
    channel_width = 2 * lr_std
    upper_band = lr_mid + channel_width
    lower_band = lr_mid - channel_width
    
    # Volume filter
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_filter = volume > (0.7 * vol_ma)
    
    # RSI for momentum confirmation
    def rsi(arr, period):
        delta = np.diff(arr, prepend=arr[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    rsi_values = rsi(close, 14)
    
    # ATR for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start = max(lr_period, 20, 14)  # Warmup
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(lr_mid[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(rsi_values[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Position management
        if position == 1:  # long position
            # Exit: price crosses below midpoint OR RSI overbought OR stoploss
            if (close[i] < lr_mid[i] or rsi_values[i] > 70 or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above midpoint OR RSI oversold OR stoploss
            if (close[i] > lr_mid[i] or rsi_values[i] < 30 or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price near bands + RSI + trend filter + volume
            near_lower = close[i] <= lower_band[i] * 1.01  # Within 1% of lower band
            near_upper = close[i] >= upper_band[i] * 0.99  # Within 1% of upper band
            rsi_oversold = rsi_values[i] < 35
            rsi_overbought = rsi_values[i] > 65
            uptrend = close[i] > ema50_1d_aligned[i]
            downtrend = close[i] < ema50_1d_aligned[i]
            
            long_setup = near_lower and rsi_oversold and uptrend and vol_filter[i]
            short_setup = near_upper and rsi_overbought and downtrend and vol_filter[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            elif short_setup:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals