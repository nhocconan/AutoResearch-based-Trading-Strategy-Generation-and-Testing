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
    
    # Get 1w data for weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA(34) on 1w close
    close_1w = df_1w['close'].values
    ema_period = 34
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period-1] = np.mean(close_1w[:ema_period])
        multiplier = 2 / (ema_period + 1)
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Align 1w EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate Donchian(20) breakout on daily data
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(donchian_period-1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # Calculate ATR(14) for volatility filter
    atr_period = 14
    tr = np.maximum(np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1])), np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[1:atr_period+1])
        for i in range(atr_period, n):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0
    size = 0.25  # 25% position size
    
    # Warmup: need Donchian (20), EMA (34), ATR (14)
    start_idx = max(donchian_period-1, ema_period, atr_period)
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_1w_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Trend filter: price above/below 1w EMA(34)
        uptrend = price > ema_1w_aligned[i]
        downtrend = price < ema_1w_aligned[i]
        
        # Volatility filter: ATR > 0.5 * price (avoid low volatility chop)
        volatility_filter = atr[i] > 0.005 * price
        
        if position == 0:
            # Long entry: price breaks above Donchian high in uptrend with volatility
            if price > donchian_high[i] and uptrend and volatility_filter:
                signals[i] = size
                position = 1
            # Short entry: price breaks below Donchian low in downtrend with volatility
            elif price < donchian_low[i] and downtrend and volatility_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend reverses
            if price < donchian_low[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend reverses
            if price > donchian_high[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_1wEMA34_VolatilityFilter"
timeframe = "1d"
leverage = 1.0