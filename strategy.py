#!/usr/bin/env python3
# 4h_1d_Momentum_Regime_Switch
# Hypothesis: Combines 4h momentum (RSI) with 1d regime detection (ADX) and volume confirmation.
# In trending regimes (ADX > 25), follows momentum (RSI > 55 for long, < 45 for short).
# In ranging regimes (ADX <= 25), mean-reverts at Bollinger Bands (2 std).
# Volume filter ensures trades occur only with institutional participation.
# Designed to work in both bull (trend following) and bear (mean reversion in ranges) markets.
# Target: 20-40 trades/year to minimize fee drag.

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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily ADX (14-period) for regime detection
    # True Range
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    # Directional Movement
    up_move = high_1d[1:] - high_1d[:-1]
    down_move = low_1d[:-1] - low_1d[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    # Smoothed values (Wilder's smoothing)
    def wilders_smoothing(data, period):
        result = np.full_like(data, np.nan)
        if len(data) >= period:
            result[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    tr14 = wilders_smoothing(tr, 14)
    plus_dm14 = wilders_smoothing(plus_dm, 14)
    minus_dm14 = wilders_smoothing(minus_dm, 14)
    
    # DI and DX
    plus_di14 = 100 * plus_dm14 / tr14
    minus_di14 = 100 * minus_dm14 / tr14
    dx = 100 * np.abs(plus_di14 - minus_di14) / (plus_di14 + minus_di14)
    adx = wilders_smoothing(dx, 14)
    
    # Align ADX to 4h
    adx_4h = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate 4h RSI (14-period)
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    def rsi_wilder(gain, loss, period):
        rsi = np.full_like(close, np.nan)
        if len(gain) >= period:
            avg_gain = np.mean(gain[:period])
            avg_loss = np.mean(loss[:period])
            rs = avg_gain / avg_loss if avg_loss != 0 else 0
            rsi[period] = 100 - (100 / (1 + rs))
            for i in range(period+1, len(close)):
                avg_gain = (avg_gain * (period-1) + gain[i-1]) / period
                avg_loss = (avg_loss * (period-1) + loss[i-1]) / period
                rs = avg_gain / avg_loss if avg_loss != 0 else 0
                rsi[i] = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = rsi_wilder(gain, loss, 14)
    
    # Calculate 4h Bollinger Bands (20-period, 2 std)
    def bollinger_bands(series, period, std_dev):
        sma = np.full_like(series, np.nan)
        std = np.full_like(series, np.nan)
        upper = np.full_like(series, np.nan)
        lower = np.full_like(series, np.nan)
        
        for i in range(period-1, len(series)):
            sma[i] = np.mean(series[i-period+1:i+1])
            std[i] = np.std(series[i-period+1:i+1])
            upper[i] = sma[i] + std_dev * std[i]
            lower[i] = sma[i] - std_dev * std[i]
        return upper, lower
    
    bb_upper, bb_lower = bollinger_bands(close, 20, 2.0)
    
    # Calculate 4h volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    for i in range(19, n):
        volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):  # Start after warmup period
        # Skip if any critical data is NaN
        if (np.isnan(adx_4h[i]) or
            np.isnan(rsi[i]) or
            np.isnan(bb_upper[i]) or
            np.isnan(bb_lower[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume > 1.3x 20-period MA
        if volume[i] < 1.3 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        # Regime detection: ADX > 25 = trending, ADX <= 25 = ranging
        is_trending = adx_4h[i] > 25
        
        if position == 0:
            if is_trending:
                # Trend following: RSI > 55 for long, < 45 for short
                if rsi[i] > 55:
                    position = 1
                    signals[i] = position_size
                elif rsi[i] < 45:
                    position = -1
                    signals[i] = -position_size
            else:
                # Mean reversion: Buy at lower BB, sell at upper BB
                if close[i] <= bb_lower[i]:
                    position = 1
                    signals[i] = position_size
                elif close[i] >= bb_upper[i]:
                    position = -1
                    signals[i] = -position_size
        elif position == 1:
            # Exit long: RSI < 45 in trend, or price > middle BB in range
            if is_trending:
                if rsi[i] < 45:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
            else:
                # Exit at middle Bollinger Band (20-period SMA)
                middle_band = np.mean(close[i-19:i+1]) if i >= 19 else close[i]
                if close[i] >= middle_band:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = position_size
        elif position == -1:
            # Exit short: RSI > 55 in trend, or price < middle BB in range
            if is_trending:
                if rsi[i] > 55:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
            else:
                # Exit at middle Bollinger Band (20-period SMA)
                middle_band = np.mean(close[i-19:i+1]) if i >= 19 else close[i]
                if close[i] <= middle_band:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -position_size
    
    return signals

name = "4h_1d_Momentum_Regime_Switch"
timeframe = "4h"
leverage = 1.0