#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_12h
Hypothesis: Trade breakouts of 20-period Donchian channels on 4h with 12h trend filter and volume confirmation.
In uptrend (price > 12h EMA34), buy when price breaks above upper Donchian with volume spike.
In downtrend (price < 12h EMA34), sell when price breaks below lower Donchian with volume spike.
Uses ATR-based stoploss to limit drawdown. Designed for 4h timeframe to target 20-50 trades/year.
Works in bull markets by capturing breakouts and in bear markets by capturing breakdowns with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(close, period):
    """Calculate Exponential Moving Average"""
    ema = np.zeros_like(close)
    if len(close) >= period:
        ema[period-1] = np.mean(close[:period])
        multiplier = 2 / (period + 1)
        for i in range(period, len(close)):
            ema[i] = (close[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema

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
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # 12h EMA34 for trend filter
    ema34_12h = calculate_ema(close_12h, 34)
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # ATR for volatility filter and stoploss
    atr = calculate_atr(prices['high'].values, prices['low'].values, prices['close'].values, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if np.isnan(ema34_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only (avoid low-volume Asian session)
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        in_session = 8 <= hour <= 20
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        atr_val = atr[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Donchian channels (20-period)
        if i >= 20:
            high_20 = prices['high'].iloc[i-20:i].max()
            low_20 = prices['low'].iloc[i-20:i].min()
        else:
            high_20 = prices['high'].iloc[:i+1].max()
            low_20 = prices['low'].iloc[:i+1].min()
        
        if position == 0:
            # Uptrend: price > 12h EMA34
            if price > ema34_12h_aligned[i]:
                # Long: price breaks above upper Donchian with volume confirmation
                if price > high_20 and volume_ok:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
            # Downtrend: price < 12h EMA34
            elif price < ema34_12h_aligned[i]:
                # Short: price breaks below lower Donchian with volume confirmation
                if price < low_20 and volume_ok:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
        
        elif position == 1:
            # Long: hold or exit
            # Exit conditions: trend reversal, stoploss, or opposite Donchian break
            if (price < ema34_12h_aligned[i] or  # trend reversal
                price < entry_price - 2.0 * atr_val or  # stoploss
                price < low_20):  # opposite Donchian break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: hold or exit
            # Exit conditions: trend reversal, stoploss, or opposite Donchian break
            if (price > ema34_12h_aligned[i] or  # trend reversal
                price > entry_price + 2.0 * atr_val or  # stoploss
                price > high_20):  # opposite Donchian break
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_12h"
timeframe = "4h"
leverage = 1.0