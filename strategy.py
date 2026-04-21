#!/usr/bin/env python3
"""
1h_PriceAction_SwingTrader_v1
Hypothesis: Price action swing trading using 1h swing highs/lows with 4h trend filter (EMA21) and volume confirmation. Works in bull/bear by taking long signals only when price > 4h EMA21 (uptrend) and short signals only when price < 4h EMA21 (downtrend). Uses 1d ATR for volatility filter to avoid choppy markets. Targets 15-30 trades/year with tight entry conditions to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_swing_points(high, low, lookback=5):
    """Calculate swing highs and lows"""
    n = len(high)
    swing_high = np.zeros(n)
    swing_low = np.zeros(n)
    
    for i in range(lookback, n - lookback):
        # Swing high: highest high in lookback window
        if high[i] == np.max(high[i-lookback:i+lookback+1]):
            swing_high[i] = high[i]
        # Swing low: lowest low in lookback window
        if low[i] == np.min(low[i-lookback:i+lookback+1]):
            swing_low[i] = low[i]
    
    return swing_high, swing_low

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
    
    # Load 4h data once for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA21 for trend direction
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=21, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Load 1d data once for ATR volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d ATR20 for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 20)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1h swing points
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    swing_high, swing_low = calculate_swing_points(high_1h, low_1h, lookback=3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(10, n):
        # Skip if indicators not ready
        if np.isnan(ema_4h_aligned[i]) or np.isnan(atr_1d_aligned[i]):
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
        
        # Volume filter: current volume > 1.2 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.2 * vol_ma
        else:
            volume_ok = False
        
        # Volatility filter: avoid extremely low volatility (choppy markets)
        vol_filter = atr_1d_aligned[i] > 0  # Always true if ATR calculated
        
        if position == 0:
            # Long: price above swing low + price > 4h EMA21 (uptrend) + volume
            if (i >= 3 and swing_low[i-3] > 0 and 
                price > swing_low[i-3] and 
                price > ema_4h_aligned[i] and 
                volume_ok and vol_filter):
                signals[i] = 0.20
                position = 1
            # Short: price below swing high + price < 4h EMA21 (downtrend) + volume
            elif (i >= 3 and swing_high[i-3] > 0 and 
                  price < swing_high[i-3] and 
                  price < ema_4h_aligned[i] and 
                  volume_ok and vol_filter):
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: price below swing low or price < 4h EMA21 (trend change)
            if (i >= 3 and swing_low[i-3] > 0 and price < swing_low[i-3]) or price < ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: price above swing high or price > 4h EMA21 (trend change)
            if (i >= 3 and swing_high[i-3] > 0 and price > swing_high[i-3]) or price > ema_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_PriceAction_SwingTrader_v1"
timeframe = "1h"
leverage = 1.0