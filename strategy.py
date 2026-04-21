#!/usr/bin/env python3
"""
6h_ElderRay_1d_Trend_Reversal_v1
Hypothesis: Elder Ray (Bull Power/Bear Power) with 1d trend filter and 1d ATR volatility filter.
- Bull Power = High - EMA13; Bear Power = EMA13 - Low
- Long when Bull Power > 0 and Bear Power < 0 (strong bullish momentum) + price > 1d EMA50 (uptrend)
- Short when Bear Power > 0 and Bull Power < 0 (strong bearish momentum) + price < 1d EMA50 (downtrend)
- Exit when momentum weakens (Bull Power < 0 for long, Bear Power < 0 for short) or trend changes
- Uses 1d ATR20 for volatility filter to avoid choppy markets
- Targets 15-30 trades/year with tight entry conditions to minimize fee drag
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(values, period):
    """Calculate Exponential Moving Average"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    ema = np.zeros_like(values)
    multiplier = 2 / (period + 1)
    ema[0] = values[0]
    for i in range(1, len(values)):
        ema[i] = (values[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

def calculate_elder_ray(high, low, close, ema_period=13):
    """Calculate Elder Ray: Bull Power and Bear Power"""
    ema = calculate_ema(close, ema_period)
    bull_power = high - ema
    bear_power = ema - low
    return bull_power, bear_power

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= period:
        atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, len(tr)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Load 1d data once for EMA trend filter and ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = df_1d['close'].values
    ema_1d_50 = calculate_ema(close_1d, 50)
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    # Calculate 1d ATR20 for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_arr = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d_arr, 20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h Elder Ray
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    bull_power, bear_power = calculate_elder_ray(high_6h, low_6h, close_6h, 13)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for indicators
        # Skip if indicators not ready
        if np.isnan(ema_1d_50_aligned[i]) or np.isnan(atr_1d_aligned[i]) or \
           np.isnan(bull_power[i]) or np.isnan(bear_power[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.3 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.3 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d_aligned[i] > 0  # Always true if ATR calculated
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (strong bullish) + price > 1d EMA50 (uptrend) + volume
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                price > ema_1d_50_aligned[i] and 
                volume_ok and vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power > 0 AND Bull Power < 0 (strong bearish) + price < 1d EMA50 (downtrend) + volume
            elif (bear_power[i] > 0 and bull_power[i] < 0 and 
                  price < ema_1d_50_aligned[i] and 
                  volume_ok and vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power < 0 (momentum weakening) or price < 1d EMA50 (trend change)
            if bull_power[i] < 0 or price < ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power < 0 (momentum weakening) or price > 1d EMA50 (trend change)
            if bear_power[i] < 0 or price > ema_1d_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1d_Trend_Reversal_v1"
timeframe = "6h"
leverage = 1.0