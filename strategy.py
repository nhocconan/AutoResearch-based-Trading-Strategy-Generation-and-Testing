#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA(20) for trend filter
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate 1d ATR(14) for volatility filter and position sizing
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 24-period average volume for confirmation (1d * 24 = 24h = 1 day)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or 
            np.isnan(atr[i]) or 
            np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # ATR-based volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.003  # Minimum 0.3% ATR relative to price
        
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = vol > (vol_avg[i] * 1.5) if not np.isnan(vol_avg[i]) else False
        
        # Trend filter: price > 1d EMA20 for long, price < 1d EMA20 for short
        trend_filter_long = price > ema_20_1d_aligned[i]
        trend_filter_short = price < ema_20_1d_aligned[i]
        
        if position == 0:
            # Long setup: price above 1d EMA20 + volume confirmation + volatility filter
            if (trend_filter_long and vol_confirm and vol_filter):
                position = 1
                signals[i] = position_size
            # Short setup: price below 1d EMA20 + volume confirmation + volatility filter
            elif (trend_filter_short and vol_confirm and vol_filter):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 1d EMA20
            if price < ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above 1d EMA20
            if price > ema_20_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1dEMA20_Volume_Filter"
timeframe = "4h"
leverage = 1.0