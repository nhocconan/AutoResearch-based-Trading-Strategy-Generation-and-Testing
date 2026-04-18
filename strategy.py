#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mlt_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel and EMA
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on daily
    upper_channel = np.full_like(close_1d, np.nan)
    lower_channel = np.full_like(close_1d, np.nan)
    
    for i in range(19, len(close_1d)):
        upper_channel[i] = np.max(high_1d[i-19:i+1])
        lower_channel[i] = np.min(low_1d[i-19:i+1])
    
    # Calculate 50-period EMA on weekly for trend filter
    if len(close_1w) >= 50:
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Calculate ATR on daily for volatility filter
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Align all data to daily timeframe (primary)
    upper_channel_daily = align_htf_to_ltf(prices, df_1d, upper_channel)
    lower_channel_daily = align_htf_to_ltf(prices, df_1d, lower_channel)
    ema_1w_daily = align_htf_to_ltf(prices, df_1w, ema_1w)
    atr_1d_daily = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(19, 50, 14) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_channel_daily[i]) or np.isnan(lower_channel_daily[i]) or 
            np.isnan(ema_1w_daily[i]) or np.isnan(atr_1d_daily[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely low volatility
        vol_filter = atr_1d_daily[i] > 0.005 * close[i]  # ATR > 0.5% of price
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with weekly uptrend
            if close[i] > upper_channel_daily[i] and close[i] > ema_1w_daily[i] and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian channel with weekly downtrend
            elif close[i] < lower_channel_daily[i] and close[i] < ema_1w_daily[i] and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian channel OR weekly trend turns down
            if close[i] < lower_channel_daily[i] or close[i] < ema_1w_daily[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian channel OR weekly trend turns up
            if close[i] > upper_channel_daily[i] or close[i] > ema_1w_daily[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA_TrendFilter"
timeframe = "1d"
leverage = 1.0