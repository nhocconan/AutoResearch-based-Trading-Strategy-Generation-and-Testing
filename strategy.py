#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Elder Ray + 12h ADX regime filter
    # Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
    # Long when Bull Power > 0 AND Bear Power < 0 AND 12h ADX > 25 (strong trend)
    # Short when Bear Power > 0 AND Bull Power < 0 AND 12h ADX > 25
    # Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 12h data for ADX regime filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate 12h ADX(14)
    def calculate_adx(high, low, close, period=14):
        # True Range
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])  # Align with original indices
        
        # Directional Movement
        up_move = high[1:] - high[:-1]
        down_move = low[:-1] - low[1:]
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
        plus_dm = np.concatenate([[0.0], plus_dm])
        minus_dm = np.concatenate([[0.0], minus_dm])
        
        # Smoothed values using Wilder's smoothing (EMA with alpha=1/period)
        def wilders_smoothing(data, period):
            alpha = 1.0 / period
            result = np.full_like(data, np.nan)
            # First value is simple average
            if len(data) >= period:
                result[period-1] = np.nanmean(data[:period])
            # Subsequent values
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]) and not np.isnan(data[i]):
                    result[i] = (1 - alpha) * result[i-1] + alpha * data[i]
            return result
        
        atr = wilders_smoothing(tr, period)
        plus_dm_smooth = wilders_smoothing(plus_dm, period)
        minus_dm_smooth = wilders_smoothing(minus_dm, period)
        
        # Avoid division by zero
        plus_di = np.where(atr != 0, 100 * plus_dm_smooth / atr, 0)
        minus_di = np.where(atr != 0, 100 * minus_dm_smooth / atr, 0)
        
        dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
        adx = wilders_smoothing(dx, period)
        return adx
    
    adx_12h = calculate_adx(high_12h, low_12h, close_12h, 14)
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Calculate EMA13 for Elder Ray on 6h
    close_series = pd.Series(close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = ema13 - low   # Bear Power = EMA13 - Low
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(adx_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: 12h ADX > 25 indicates strong trend
        strong_trend = adx_12h_aligned[i] > 25
        
        # Elder Ray signals
        bull_positive = bull_power[i] > 0
        bear_positive = bear_power[i] > 0
        
        # Entry conditions
        long_entry = bull_positive and not bear_positive and strong_trend
        short_entry = bear_positive and not bull_positive and strong_trend
        
        # Exit conditions: when Elder Ray signals reverse or trend weakens
        long_exit = not bull_positive or bear_positive or not strong_trend
        short_exit = not bear_positive or bull_positive or not strong_trend
        
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
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_12h_elder_ray_adx_regime_v1"
timeframe = "6h"
leverage = 1.0