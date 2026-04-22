#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4h data once for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Load 1d data for ATR-based volatility filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1d ATR(14) for volatility filter
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([np.array([np.nan]), tr1])
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # 1h ATR(14) for position sizing and stops
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([np.array([np.nan]), tr])
    atr14_1h = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any data is not ready
        if (np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or 
            np.isnan(atr14_1h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        atr1h = atr14_1h[i]
        ema50 = ema50_4h_aligned[i]
        atr1d = atr14_1d_aligned[i]
        
        # Dynamic position size based on volatility (inverse volatility scaling)
        base_size = 0.20
        vol_scaling = min(2.0, max(0.5, atr1d / (np.nanmedian(atr14_1d_aligned[:i+1]) + 1e-10)))
        position_size = base_size / vol_scaling
        position_size = min(0.35, max(0.10, position_size))  # clamp between 0.10 and 0.35
        
        if position == 0:
            # Long: price > 4h EMA50 and momentum breakout
            if price > ema50 and price > close[i-1] + 0.5 * atr1h:
                signals[i] = position_size
                position = 1
                entry_price = price
            # Short: price < 4h EMA50 and momentum breakdown
            elif price < ema50 and price < close[i-1] - 0.5 * atr1h:
                signals[i] = -position_size
                position = -1
                entry_price = price
        
        elif position != 0:
            # Stop loss: 2.0 * ATR(1h)
            stop_distance = 2.0 * atr1h
            
            if position == 1:  # long position
                if price <= entry_price - stop_distance:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:  # short position
                if price >= entry_price + stop_distance:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals

# Strategy designed for 1h timeframe with 4h trend filter and volatility-adjusted sizing
# Works in both bull and bear markets by following 4h trend while using volatility breaks for entry
# Low trade frequency target: 15-35 trades/year to minimize fee drag
name = "1h_EMA50_Trend_VolatilityBreak"
timeframe = "1h"
leverage = 1.0