#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA trend filter and ATR-based position sizing.
Long when price breaks above 20-day high with 1w EMA(50) confirming uptrend.
Short when price breaks below 20-day low with 1w EMA(50) confirming downtrend.
Position size scaled by inverse ATR volatility (0.25 max size) to reduce drawdown.
Uses discrete position sizes (0.0, ±0.25) to minimize fee churn.
Target: 30-100 total trades over 4 years (7-25/year).
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1d Donchian channels (20-period)
    def calculate_donchian(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        return upper, lower
    
    donchian_upper_1d, donchian_lower_1d = calculate_donchian(high_1d, low_1d, 20)
    
    # Calculate 1w EMA(50) for trend filter
    def calculate_ema_close(close, span=50):
        ema = np.full_like(close, np.nan)
        if len(close) >= span:
            multiplier = 2 / (span + 1)
            ema[span-1] = np.mean(close[:span])
            for i in range(span, len(close)):
                ema[i] = (close[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        return ema
    
    ema_50_1w = calculate_ema_close(close_1w, 50)
    
    # Calculate 1d ATR(14) for volatility-based position sizing
    def calculate_atr(high, low, close, period=14):
        tr = np.zeros_like(close)
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        # Wilder's ATR
        if len(tr) >= period:
            atr[period] = np.mean(tr[1:period+1])
            for i in range(period+1, len(tr)):
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align indicators to primary timeframe (1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper_1d)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_upper_aligned[i]
        lower = donchian_lower_aligned[i]
        ema_trend = ema_50_1w_aligned[i]
        atr_val = atr_14_1d_aligned[i]
        
        # Avoid division by zero
        if atr_val <= 0:
            atr_val = 0.001
        
        # Volatility-adjusted position size (max 0.25)
        # Size inversely proportional to ATR (higher volatility = smaller position)
        base_size = 0.25
        vol_factor = min(2.0, max(0.5, 0.01 / atr_val))  # normalize ATR
        pos_size = base_size * vol_factor
        pos_size = min(0.25, max(0.05, pos_size))  # clamp between 0.05 and 0.25
        
        if position == 0:
            # Long: price breaks above Donchian upper with 1w EMA uptrend
            if price > upper and price > ema_trend:
                signals[i] = pos_size
                position = 1
            # Short: price breaks below Donchian lower with 1w EMA downtrend
            elif price < lower and price < ema_trend:
                signals[i] = -pos_size
                position = -1
        
        elif position == 1:
            # Exit long: price returns to Donchian lower OR trend reverses
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = pos_size
        
        elif position == -1:
            # Exit short: price returns to Donchian upper OR trend reverses
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -pos_size
    
    return signals

name = "1d_Donchian20_1wEMA50_ATRSize"
timeframe = "1d"
leverage = 1.0