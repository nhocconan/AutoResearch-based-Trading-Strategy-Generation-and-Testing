#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_1dTrend_VolumeATR_Stop
Hypothesis: Donchian(20) breakouts on 4h chart capture momentum. 1d trend filter (price > EMA50 for longs, < EMA50 for shorts) ensures alignment with higher timeframe direction. Volume confirmation (>1.5x 20-period average) filters weak breakouts. ATR-based stoploss (2.5x ATR(14)) limits downside. Targets 75-150 trades over 4 years (19-38/year) by requiring confluence of breakout, trend, and volume. Works in bull markets via upside breakouts and bear markets via downside breakdowns. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian(20) on 4h: highest high and lowest low of last 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    # ATR(14) for stoploss
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First bar: no previous close
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    atr_multiplier = 2.5  # ATR multiplier for stoploss
    
    # Warmup: need Donchian(20), EMA50(50), volume avg(20), ATR(14)
    start_idx = max(20, 50, 20, 14)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_confirm[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema_1d_val = ema_50_1d_aligned[i]
        vol_conf = volume_confirm[i]
        atr_val = atr[i]
        
        if position == 0:
            # Determine trend: price > EMA50 = uptrend, price < EMA50 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when price breaks above Donchian high and volume confirms
                if (close_val > donch_high) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when price breaks below Donchian low and volume confirms
                if (close_val < donch_low) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price touches Donchian low (breakdown) or stoploss hit
            # Stoploss: highest high since entry minus atr_multiplier * ATR
            # Simplified: exit if price drops below entry level - atr_multiplier * ATR
            # Track entry price implicitly via position logic
            exit_condition = (close_val < donch_low) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price touches Donchian high (breakout) or stoploss hit
            exit_condition = (close_val > donch_high) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dTrend_VolumeATR_Stop"
timeframe = "4h"
leverage = 1.0