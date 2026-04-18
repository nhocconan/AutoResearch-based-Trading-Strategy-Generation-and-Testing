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
    
    # Get daily data for Bollinger Bands and ATR
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period Bollinger Bands (std=2)
    def calculate_bollinger_bands(close, period=20, std_dev=2):
        if len(close) < period:
            upper = np.full_like(close, np.nan)
            lower = np.full_like(close, np.nan)
            middle = np.full_like(close, np.nan)
            return upper, middle, lower
        
        # Calculate SMA and std
        sma = np.full_like(close, np.nan)
        std = np.full_like(close, np.nan)
        
        for i in range(period-1, len(close)):
            sma[i] = np.mean(close[i-period+1:i+1])
            std[i] = np.std(close[i-period+1:i+1])
        
        upper = sma + std_dev * std
        lower = sma - std_dev * std
        middle = sma
        
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
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(close_1d, 20, 2)
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
    bb_upper_12h = align_htf_to_ltf(prices, df_1d, bb_upper)
    bb_lower_12h = align_htf_to_ltf(prices, df_1d, bb_lower)
    bb_middle_12h = align_htf_to_ltf(prices, df_1d, bb_middle)
    atr_12h = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1w_12h = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Volume confirmation: volume > 2.0x 12-period average
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
        if (np.isnan(bb_upper_12h[i]) or np.isnan(bb_lower_12h[i]) or 
            np.isnan(bb_middle_12h[i]) or np.isnan(atr_12h[i]) or 
            np.isnan(ema_1w_12h[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Bollinger Band squeeze detection (width < 50% of 20-period average)
        bb_width = bb_upper_12h[i] - bb_lower_12h[i]
        if i >= 20:
            width_ma = np.mean(bb_upper_12h[i-19:i+1] - bb_lower_12h[i-19:i+1])
            squeeze = bb_width < 0.5 * width_ma
        else:
            squeeze = False
        
        if position == 0:
            # Long: price breaks above upper BB with volume in squeeze condition
            if close[i] > bb_upper_12h[i] and vol_confirm and squeeze:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower BB with volume in squeeze condition
            elif close[i] < bb_lower_12h[i] and vol_confirm and squeeze:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below middle BB OR volatility expands (width > 2x average)
            if close[i] < bb_middle_12h[i] or bb_width > 2.0 * width_ma:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above middle BB OR volatility expands
            if close[i] > bb_middle_12h[i] or bb_width > 2.0 * width_ma:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Bollinger_Squeeze_Volume_Reversal"
timeframe = "12h"
leverage = 1.0