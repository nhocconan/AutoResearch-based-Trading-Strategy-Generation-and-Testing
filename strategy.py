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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly Donchian channels (20-period)
    high_20w = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    donch_high_20w = align_htf_to_ltf(prices, df_1w, high_20w)
    donch_low_20w = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Calculate weekly EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 6-day ATR(14) for volatility filter
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)
    tr1 = high_series - low_series
    tr2 = abs(high_series - close_series.shift(1))
    tr3 = abs(low_series - close_series.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = 100
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(donch_high_20w[i]) or 
            np.isnan(donch_low_20w[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        
        # Volatility filter: avoid extremely low volatility periods
        atr_ratio = atr[i] / price if price > 0 else 0
        vol_filter = atr_ratio > 0.005  # Minimum 0.5% ATR relative to price
        
        # Trend filter based on weekly EMA50
        trend_filter_long = price > ema_50_1w_aligned[i]
        trend_filter_short = price < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long breakout: price breaks above weekly Donchian high + volatility + trend
            if (price > donch_high_20w[i] and vol_filter and trend_filter_long):
                position = 1
                signals[i] = position_size
            # Short breakdown: price breaks below weekly Donchian low + volatility + trend
            elif (price < donch_low_20w[i] and vol_filter and trend_filter_short):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly Donchian low (trailing stop)
            if price < donch_low_20w[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly Donchian high (trailing stop)
            if price > donch_high_20w[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1wDonchian20_EMA50_Trend_Breakout_v1"
timeframe = "6h"
leverage = 1.0