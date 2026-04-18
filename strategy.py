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
    
    # Get daily data for Donchian channels and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Donchian channels (upper/lower) and middle
    def calculate_donchian_channels(high, low, period=20):
        if len(high) < period:
            upper = np.full_like(high, np.nan)
            lower = np.full_like(high, np.nan)
            middle = np.full_like(high, np.nan)
            return upper, middle, lower
        
        upper = np.full_like(high, np.nan)
        lower = np.full_like(high, np.nan)
        middle = np.full_like(high, np.nan)
        
        for i in range(period-1, len(high)):
            upper[i] = np.max(high[i-period+1:i+1])
            lower[i] = np.min(low[i-period+1:i+1])
        
        middle = (upper + lower) / 2
        return upper, middle, lower
    
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
    
    donch_upper, donch_middle, donch_lower = calculate_donchian_channels(high_1d, low_1d, 20)
    atr_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA(34) for trend filter
    if len(close_1w) >= 34:
        ema_1w = pd.Series(close_1w).ewm(span=34, adjust=False).mean().values
    else:
        ema_1w = np.full_like(close_1w, np.nan)
    
    # Align all data to 12h timeframe
    donch_upper_12h = align_htf_to_ltf(prices, df_1d, donch_upper)
    donch_lower_12h = align_htf_to_ltf(prices, df_1d, donch_lower)
    donch_middle_12h = align_htf_to_ltf(prices, df_1d, donch_middle)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 1.5x 12-period average
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 12
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14, 34) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_upper_12h[i]) or np.isnan(donch_lower_12h[i]) or 
            np.isnan(donch_middle_12h[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below weekly EMA34
        price_above_weekly_ema = close[i] > ema_1w_12h[i]
        price_below_weekly_ema = close[i] < ema_1w_12h[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and above weekly EMA
            if close[i] > donch_upper_12h[i] and vol_confirm and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and below weekly EMA
            elif close[i] < donch_lower_12h[i] and vol_confirm and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle Donchian OR volatility expands (ATR > 1.5x average)
            if close[i] < donch_middle_12h[i] or atr_12h[i] > 1.5 * np.nanmean(atr_12h[max(0, i-20):i]):
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle Donchian OR volatility expands
            if close[i] > donch_middle_12h[i] or atr_12h[i] > 1.5 * np.nanmean(atr_12h[max(0, i-20):i]):
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_WeeklyEMA_Volume"
timeframe = "12h"
leverage = 1.0