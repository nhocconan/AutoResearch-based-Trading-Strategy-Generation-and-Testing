#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Donchian channel and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channel
    def calculate_donchian_channel(high, low, period=20):
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        
        return upper, lower
    
    # Calculate 14-period ATR
    def calculate_atr(high, low, close, period=14):
        if len(high) < period + 1:
            return np.full_like(high, np.nan)
        
        # True Range
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder smoothing for ATR
        atr = np.full_like(high, np.nan)
        atr[period] = np.mean(tr[1:period+1])
        for i in range(period+1, len(high)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to 4h timeframe
    donch_upper_4h = align_htf_to_ltf(prices, df_1d, calculate_donchian_channel(high_1d, low_1d, 20)[0])
    donch_lower_4h = align_htf_to_ltf(prices, df_1d, calculate_donchian_channel(high_1d, low_1d, 20)[1])
    atr_4h = align_htf_to_ltf(prices, df_1d, calculate_atr(high_1d, low_1d, close_1d, 14))
    ema_1w_4h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.8x 20-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 20
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_4h[i]) or np.isnan(donch_lower_4h[i]) or 
            np.isnan(atr_4h[i]) or np.isnan(ema_1w_4h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and bullish weekly trend
            if close[i] > donch_upper_4h[i] and vol_confirm and close[i] > ema_1w_4h[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and bearish weekly trend
            elif close[i] < donch_lower_4h[i] and vol_confirm and close[i] < ema_1w_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below lower Donchian OR trailing stop (2*ATR below high)
            if close[i] < donch_lower_4h[i]:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above upper Donchian OR trailing stop (2*ATR above low)
            if close[i] > donch_upper_4h[i]:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_WeeklyTrend_Volume_Filter"
timeframe = "4h"
leverage = 1.0